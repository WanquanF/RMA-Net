#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "util_3drotation_log_exp.h"
#include <sys/stat.h>
#include <sys/types.h> 

struct TriTraits : public OpenMesh::DefaultTraits
{
  /// Use double precision points
  typedef OpenMesh::Vec3d Point;
  /// Use double precision Normals
  typedef OpenMesh::Vec3d Normal;
  /// Use double precision TexCood2D
  typedef OpenMesh::Vec2d TexCoord2D;

  /// Use RGBA Color
  typedef OpenMesh::Vec4f Color;

    /// Status
    VertexAttributes(OpenMesh::Attributes::Status);
    FaceAttributes(OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);

};

/// Simple Name for Mesh
typedef OpenMesh::TriMesh_ArrayKernelT<TriTraits>  TriMesh;

class Generate_ACAP{
public:
    int nver;
    int nhalfedge;

    Generate_ACAP(std::string source_mesh_file, std::string target_mesh_file);
    void record_logR_S(std::string record_file);

    void compute_AtA();
    void reconstruction(Eigen::VectorXd feature, double alpha, std::string output_file); // TriMesh &mesh, const Eigen::Vector3d &pos, int id);

private:
    TriMesh source_mesh_;
    TriMesh target_mesh_;

    Eigen::MatrixXd logR_S_representation;
    OpenMesh::EPropHandleT<double> LB_weights;
    OpenMesh::VPropHandleT<Eigen::Matrix3d> T_matrixs;
    OpenMesh::VPropHandleT<Eigen::Matrix3d> T_w_matrix;

    Eigen::SparseMatrix<double> A_;
    Eigen::SparseMatrix<double> AtA_;  // At*A
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> AtA_solver_;
    // Eigen::SparseLU<Eigen::SparseMatrix<double> > A_solver_;
    OpenMesh::VPropHandleT<Eigen::Matrix3d> rotation_matrixs;
    OpenMesh::VPropHandleT<Eigen::Matrix3d> scaling_matrixs;

private:
    void read_source_mesh(std::string mesh_file);
    void compute_LB_weight();
    void read_target_mesh(std::string mesh_file);
    void compute_Ti(TriMesh::VertexHandle v_it,TriMesh::VertexHandle v_to_it);
    void compute_feature();
};

Generate_ACAP::Generate_ACAP(std::string source_mesh_file, std::string target_mesh_file)
{
    source_mesh_.add_property(LB_weights);
    source_mesh_.add_property(rotation_matrixs);
    source_mesh_.add_property(scaling_matrixs);
    target_mesh_.add_property(T_matrixs);

    read_source_mesh(source_mesh_file);

    read_target_mesh(target_mesh_file);
}

void Generate_ACAP::read_source_mesh(std::string mesh_file)
{
    if(!OpenMesh::IO::read_mesh(source_mesh_, mesh_file))
    {
        std::cout << "read_source_mesh :: read file wrong!" << std::endl;
    }
    std::cout << "vertex number: " << source_mesh_.n_vertices() << std::endl;
    std::cout << "halfedge number: " << source_mesh_.n_halfedges() << std::endl;
    nver = source_mesh_.n_vertices();
    nhalfedge = source_mesh_.n_vertices();

    compute_LB_weight();
}

void Generate_ACAP::compute_LB_weight()
{
    TriMesh::EdgeIter e_it, e_end(source_mesh_.edges_end());
    TriMesh::HalfedgeHandle    h0, h1, h2;
    TriMesh::VertexHandle      v0, v1;
    TriMesh::Point             p0, p1, p2, d0, d1;
    TriMesh::Scalar w;

    for (e_it = source_mesh_.edges_begin(); e_it != e_end; e_it++)
    {
        w  = 0.0;
        if(source_mesh_.is_boundary(*e_it))
        {
            h0 = source_mesh_.halfedge_handle(e_it.handle(), 0);
            if(source_mesh_.is_boundary(h0))
                h0 = source_mesh_.opposite_halfedge_handle(h0);

            v0 = source_mesh_.to_vertex_handle(h0);
            v1 = source_mesh_.from_vertex_handle(h0);
            p0 = source_mesh_.point(v0);
            p1 = source_mesh_.point(v1);
            h1 = source_mesh_.next_halfedge_handle(h0);
            p2 = source_mesh_.point(source_mesh_.to_vertex_handle(h1));
            d0 = (p0 - p2).normalize();
            d1 = (p1 - p2).normalize();
            w += 2.0 / tan(acos(std::min(0.99, std::max(-0.99, (d0|d1)))));

           w = std::max(0.0, w);//w小于0的时候还要仔细思考一下怎么处理
            if(std::isnan(w))
                std::cout << "Some weight NAN" << std::endl;
            source_mesh_.property(LB_weights, e_it) = w;
            continue;
        }
        h0 = source_mesh_.halfedge_handle(e_it.handle(), 0);
        v0 = source_mesh_.to_vertex_handle(h0);
        p0 = source_mesh_.point(v0);

        h1 = source_mesh_.opposite_halfedge_handle(h0);
        v1 = source_mesh_.to_vertex_handle(h1);
        p1 = source_mesh_.point(v1);

        h2 = source_mesh_.next_halfedge_handle(h0);
        p2 = source_mesh_.point(source_mesh_.to_vertex_handle(h2));
        d0 = (p0 - p2).normalize();
        d1 = (p1 - p2).normalize();
        w += 1.0/ tan(acos(std::max(-1.0, std::min(1.0, dot(d1, d0)))));

        h2 = source_mesh_.next_halfedge_handle(h1);
        p2 = source_mesh_.point(source_mesh_.to_vertex_handle(h2));
        d0 = (p0 - p2).normalize();
        d1 = (p1 - p2).normalize();
        w += 1.0 / tan(acos(std::max(-1.0, std::min(1.0, dot(d1,d0)))));

        if(std::isnan(w))
            std::cout<<"Some weight is NAN"<<std::endl;
        w = std::max(0.0, w);
        source_mesh_.property(LB_weights, e_it) = w;
    }
}

void Generate_ACAP::read_target_mesh(std::string mesh_file)
{
    logR_S_representation.resize(3, 3 * nver * 2);

    for(TriMesh::VertexIter v_it = source_mesh_.vertices_begin(); v_it != source_mesh_.vertices_end(); v_it++)
    {
        logR_S_representation.block(0, 3 * nver + 3 * (*v_it).idx(), 3, 3) = Eigen::Matrix3d::Identity();
    }

    if(!OpenMesh::IO::read_mesh(target_mesh_, mesh_file))
    {
        std::cout << "read_target_mesh :: read file wrong!" << std::endl;
    }

    TriMesh::FaceIter f_it = target_mesh_.faces_begin();
    for(; f_it != target_mesh_.faces_end(); f_it++)
    {
        TriMesh::FaceHandle f_h = *f_it;
        TriMesh::FaceVertexIter fe_it0, fe_it1;
        fe_it0 = source_mesh_.fv_iter(f_h);
        fe_it1 = target_mesh_.fv_iter(f_h);
        int v00, v01, v02, v10, v11, v12;
        v00 = (*fe_it0).idx(); fe_it0++;
        v01 = (*fe_it0).idx(); fe_it0++;
        v02 = (*fe_it0).idx(); fe_it0++;

        v10 = (*fe_it1).idx(); fe_it1++;
        v11 = (*fe_it1).idx(); fe_it1++;
        v12 = (*fe_it1).idx(); fe_it1++;

        if(v00 == v10)
        {
            if(v01 != v11 || v02 != v12)
            {
                std::cout << "RIMD defor and ref are not compatible!!!" << std::endl;
                return;
            }
        }
        else if(v00 == v11)
        {
            if(v01 != v12 || v02 != v10)
            {
                std::cout << "RIMD defor and ref are not compatible!!!" << std::endl;
                return;
            }
        }
        else if(v00 == v12)
        {
            if(v01 != v10 || v02 != v11)
            {
                std::cout << "RIMD defor and ref are not compatible!!!" << std::endl;
                return;
            }
        }
        else
        {
            std::cout << "RIMD defor and ref are not compatible!!!" << std::endl;
            return;
        }
    }

    TriMesh::VertexIter v_it, v_to_it;
    // 这里要求ref和defor的网格不仅拓扑一致，顶点的顺序也是要对应好的
    for(v_it = source_mesh_.vertices_begin(), v_to_it = target_mesh_.vertices_begin()
        ; v_it != source_mesh_.vertices_end() && v_to_it != target_mesh_.vertices_end()
        ; v_it++, v_to_it++)
    {
        if((*v_it).idx() != (*v_to_it).idx())
            std::cout << "RIMD :: compute_ref_to_defor_Tmatrixs different topology!!!" << std::endl;
        compute_Ti(*v_it, *v_to_it);
    }
    compute_feature();
}

void Generate_ACAP::compute_Ti(TriMesh::VertexHandle v_it, TriMesh::VertexHandle v_to_it)
{
    if(v_it.idx() != v_to_it.idx())
        std::cout << "compute_Ti correspond is wrong!!!" << std::endl;
    TriMesh::VertexEdgeIter veiter = source_mesh_.ve_iter(v_it);
    int v_id = v_it.idx();
    TriMesh::Point p0, p1;
    p0 = source_mesh_.point(v_it);
    p1 = target_mesh_.point(v_to_it);
    Eigen::Matrix3d L, RI;
    L.setZero();
    RI.setZero();
    double tolerance = 1.0e-6;
    TriMesh::HalfedgeHandle h_e = source_mesh_.halfedge_handle(v_it);
    TriMesh::VertexHandle test_v = source_mesh_.to_vertex_handle(h_e);
    TriMesh::Point tp0, tp1;
    tp0 = source_mesh_.point(v_it); tp1 = source_mesh_.point(test_v);
    double scale = 1.0;
    if(((tp0[0] - tp1[0]) * (tp0[0] - tp1[0]) + (tp0[1] - tp1[1]) * (tp0[1] - tp1[1]) + (tp0[2] - tp1[2]) * (tp0[2] - tp1[2])) < 0.01)
        scale = 100000;

    for(; veiter.is_valid(); veiter++)
    {
//        double weight = source_mesh_.property(LB_weights,(*veiter));
        double weight = 1.0;
        int to_id;
        TriMesh::VertexHandle to_v = source_mesh_.to_vertex_handle(source_mesh_.halfedge_handle(*veiter, 0));
        if(to_v.idx() == v_id)
            to_v = source_mesh_.from_vertex_handle(source_mesh_.halfedge_handle(*veiter, 0));
        to_id = to_v.idx();
        Eigen::Vector3d eij0, eij1;
        TriMesh::Point q0, q1;
        q0 = source_mesh_.point(to_v);
        q1 = target_mesh_.point(to_v);
        eij0(0) = p0[0] - q0[0];
        eij0(1) = p0[1] - q0[1];
        eij0(2) = p0[2] - q0[2];
        eij0 *= weight * scale;

        eij1(0) = p1[0] - q1[0];
        eij1(1) = p1[1] - q1[1];
        eij1(2) = p1[2] - q1[2];
        eij1 *= weight * scale;

        L += eij1*eij0.transpose();
        RI += eij0*eij0.transpose();
    }
    Eigen::Matrix3d T;
    if(fabs(RI.determinant()) > tolerance)
         T = L * RI.inverse();
    else
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(RI, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3d U, V;
        U = svd.matrixU();
        V = svd.matrixV();
        Eigen::Matrix3d S_inv(svd.singularValues().asDiagonal());
        for(int i = 0; i < 3; i++)
        {
            if(fabs(S_inv(i, i)) > tolerance)
                S_inv(i, i) = 1.0 / S_inv(i,i);
            else
                S_inv(i,i) = 0.0;
        }
        T = L * V * S_inv * U.transpose();
//        std::cout<<"T "<<(v_it).idx()<<" is singular, use pseudo inverse to computation!"<<std::endl;
    }
    target_mesh_.property(T_matrixs, v_it) = T;
}

void Generate_ACAP::compute_feature(){
    TriMesh::VertexIter v_it;
    for(v_it=target_mesh_.vertices_begin(); v_it!=target_mesh_.vertices_end(); v_it++)
    {
        Eigen::Matrix3d T = target_mesh_.property(T_matrixs,*v_it);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3d U,V;
        U=svd.matrixU();
        V=svd.matrixV();
        Eigen::Matrix3d S(svd.singularValues().asDiagonal());
        Eigen::Matrix3d Temp=Eigen::Matrix3d::Identity();
        Temp(2,2) = (U*V.transpose()).determinant();
        Eigen::Matrix3d R=U*Temp*V.transpose();
        Eigen::Matrix3d Scale = V*Temp*S*V.transpose();
        logR_S_representation.block(0, 3*(*v_it).idx(), 3, 3) = rotation_log_exp::log(R);
        logR_S_representation.block(0, 3*nver+3*(*v_it).idx(), 3, 3)=Scale;
    }
}

void Generate_ACAP::record_logR_S(std::string record_file)
{
    std::ofstream out_file(record_file);
    for(int i = 0; i < nver; i++)
    {
        out_file << logR_S_representation(2, 3 * i + 1) << std::endl;
        out_file << logR_S_representation(0, 3 * i + 2) << std::endl;
        out_file << logR_S_representation(1, 3 * i) << std::endl;
        out_file << logR_S_representation(0, 3 * nver + 3 * i) << std::endl;
        out_file << logR_S_representation(0, 3 * nver + 3 * i + 1) << std::endl;
        out_file << logR_S_representation(0, 3 * nver + 3 * i + 2) << std::endl;
        out_file << logR_S_representation(1, 3 * nver + 3 * i + 1) << std::endl;
        out_file << logR_S_representation(1, 3 * nver + 3 * i + 2) << std::endl;
        out_file << logR_S_representation(2, 3 * nver + 3 * i + 2) << std::endl;
    }
    out_file.close();
}

void Generate_ACAP::compute_AtA()
{
    A_.resize(3*source_mesh_.n_vertices(),3*source_mesh_.n_vertices());
    std::vector<Eigen::Triplet<double> >    tripletlist;
    // std::vector<int> debug_rows;
    // std::vector<int> debug_cols;
    // std::vector<double> debug_vs;
    for(TriMesh::VertexIter v_it = source_mesh_.vertices_begin();v_it!=source_mesh_.vertices_end();v_it++)
    {
        TriMesh::VertexEdgeIter vej_it = source_mesh_.ve_iter(*v_it);
        TriMesh::VertexHandle vi = *v_it;
        double total_weights=0.0;
        for(;vej_it.is_valid();vej_it++)
        {
            TriMesh::HalfedgeHandle h_eij=source_mesh_.halfedge_handle(*vej_it,0);
            TriMesh::VertexHandle vj = source_mesh_.to_vertex_handle(h_eij);
            if(vj.idx() == vi.idx())
            {
                vj = source_mesh_.from_vertex_handle(source_mesh_.halfedge_handle(*vej_it,0));
                h_eij = source_mesh_.opposite_halfedge_handle(h_eij);
            }

            double weight=source_mesh_.property(LB_weights,*vej_it);
            tripletlist.push_back(Eigen::Triplet<double>(3*vi.idx(),3*vj.idx(),-2*weight));
            tripletlist.push_back(Eigen::Triplet<double>(3*vi.idx()+1,3*vj.idx()+1,-2*weight));
            tripletlist.push_back(Eigen::Triplet<double>(3*vi.idx()+2,3*vj.idx()+2,-2*weight));
            total_weights+=weight;

            // debug_rows.push_back(vi.idx());
            // debug_cols.push_back(vj.idx());
            // debug_vs.push_back(-2*weight);
        }
        tripletlist.push_back(Eigen::Triplet<double>(3*vi.idx(),3*vi.idx(),2*total_weights));
        tripletlist.push_back(Eigen::Triplet<double>(3*vi.idx()+1,3*vi.idx()+1,2*total_weights));
        tripletlist.push_back(Eigen::Triplet<double>(3*vi.idx()+2,3*vi.idx()+2,2*total_weights));

        // debug_rows.push_back(vi.idx());
        // debug_cols.push_back(vi.idx());
        // debug_vs.push_back(2*total_weights);
    }

    // std::ofstream file("A_info.txt");
    // for(int i=0;i<debug_vs.size();i++)
    //     file<<debug_rows[i]<<" "<<debug_cols[i]<<" "<<debug_vs[i]<<std::endl;
    // file.close();

    A_.setFromTriplets(tripletlist.begin(),tripletlist.end());
    /*
    A_solver_.compute(A_);
    if(A_solver_.info()!=Eigen::Success) {
      // decomposition failed
        std::cout<<"A decompose solver compute error:"<<A_solver_.info()<<std::endl;
    }

    std::cout<<A_<<std::endl;
    std::cout<<"A determinant:"<<A_solver_.determinant()<<std::endl;
    */
    AtA_.resize(3*nver, 3*nver);
    AtA_ = A_.transpose()* A_;
    AtA_solver_.compute(AtA_);
    if(AtA_solver_.info()!=Eigen::Success) {
      // decomposition failed
        std::cout<<"A decompose solver compute error:"<<AtA_solver_.info()<<std::endl;
    }
}

void Generate_ACAP::reconstruction(Eigen::VectorXd feature, double alpha, std::string output_file) // , TriMesh &mesh, const Eigen::Vector3d &pos, int id)
{
    if(feature.size()!=source_mesh_.n_vertices()*9)
    {
        std::cout<<"reconstruction with wrong feature!"<<std::endl;
        return;
    }

    for(int i=0;i<source_mesh_.n_vertices();i++)
    {
        TriMesh::VertexHandle vh(i);

        Eigen::Vector3d angle_axis;
        angle_axis(0)=feature(9*i) * alpha;
        angle_axis(1)=feature(9*i+1) * alpha;
        angle_axis(2)=feature(9*i+2) * alpha;
        Eigen::Matrix3d R=rotation_log_exp::exp(angle_axis);
        source_mesh_.property(rotation_matrixs,vh)=R;

        Eigen::Matrix3d S;
        S(0,0)=feature(9*i+3);
        S(0,1)=feature(9*i+4);
        S(0,2)=feature(9*i+5);
        S(1,1)=feature(9*i+6);
        S(1,2)=feature(9*i+7);
        S(2,2)=feature(9*i+8);
        S(1,0)=S(0,1);  S(2,0)=S(0,2);  S(2,1)=S(1,2);
        source_mesh_.property(scaling_matrixs,vh)=S;

        source_mesh_.property(T_matrixs,vh)=R*S;
    }

    Eigen::VectorXd B(3*source_mesh_.n_vertices());
    B.setZero();
    for(TriMesh::VertexIter v_it = source_mesh_.vertices_begin();v_it!=source_mesh_.vertices_end();v_it++)
    {
        TriMesh::VertexEdgeIter vej_it = source_mesh_.ve_iter(*v_it);
        TriMesh::VertexHandle vi = *v_it;
        Eigen::Matrix3d Ti=source_mesh_.property(T_matrixs,vi);
        for(;vej_it.is_valid();vej_it++)
        {
            TriMesh::HalfedgeHandle h_eij=source_mesh_.halfedge_handle(*vej_it,0);
            TriMesh::VertexHandle vj = source_mesh_.to_vertex_handle(h_eij);
            if(vj.idx() == vi.idx())
            {
                vj = source_mesh_.from_vertex_handle(source_mesh_.halfedge_handle(*vej_it,0));
                h_eij = source_mesh_.opposite_halfedge_handle(h_eij);
            }
            double weight=source_mesh_.property(LB_weights,*vej_it);
            TriMesh::Point pi=source_mesh_.point(vi);
            TriMesh::Point pj=source_mesh_.point(vj);
            Eigen::Vector3d eij;
            eij(0)=pi[0]-pj[0];
            eij(1)=pi[1]-pj[1];
            eij(2)=pi[2]-pj[2];
            Eigen::Matrix3d Tj=source_mesh_.property(T_matrixs,vj);
            B.block<3,1>(3*vi.idx(),0)+=weight*(Ti+Tj)*eij;
        }
    }

    Eigen::VectorXd positions=AtA_solver_.solve(A_.transpose()*B);
    /*
    if(id!=-1)
    {
        Eigen::Vector3d translate=pos-positions.block<3,1>(3*id,0);
        for(int i=0;i<source_mesh_.n_vertices();i++)
            positions.block<3,1>(3*i,0)=positions.block<3,1>(3*i,0) +translate;
    }
    */
    Eigen::Vector3d mean_position;
    for(int i = 0; i < 3; i++)
        mean_position(i) = 0.0;
    for(int i=0;i<source_mesh_.n_vertices();i++)
        mean_position += positions.block<3,1>(3*i,0);
    mean_position /= source_mesh_.n_vertices();
    for(int i=0;i<source_mesh_.n_vertices();i++)
        positions.block<3,1>(3*i,0) -= mean_position;
    TriMesh mesh=source_mesh_;
    TriMesh::VertexIter v_it = mesh.vertices_begin();
    double *addrV = const_cast<double*>(mesh.point(*v_it).data());
    memcpy(addrV,positions.data(),sizeof(double)*positions.size());

    OpenMesh::IO::write_mesh(mesh, output_file);
}

int main(int argc, char *argv[] )
{
    /// some paths
    //std::string source_mesh_file = "/home/chr/Downloads/test_mesh/cat0.obj"; // the path of the source mesh
    //std::string target_mesh_file = "/home/chr/Downloads/test_mesh/cat3.obj"; // the path of the target mesh
    //std::string record_file = "/home/chr/Downloads/acap.txt"; // the path to record the 'acap' feature
    //std::string output_folder = "/home/chr/Downloads/insert_mesh/"; // the path of the folder to save the deformed meshes


    std::string source_mesh_file = argv[1];
    std::string target_mesh_file = argv[2];
//    std::string record_file = argv[3];
//    std::string output_folder = argv[4];
    std::string output_file = argv[3];
    
//    mkdir(output_folder.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    std::string record_file = source_mesh_file.substr(0,source_mesh_file.length()-4)+"__acap_record.txt";
    std::cout<<record_file<<std::endl;
    
//    int N = std::atoi(argv[5]);
    double alpha = std::atof(argv[4]);

    /// vertex to ACAP feature
    Generate_ACAP my_acap(source_mesh_file, target_mesh_file);
    my_acap.record_logR_S(record_file);

    /// compute A
    my_acap.compute_AtA();

    /// read ACAP
    Eigen::VectorXd feature;
    feature.resize(9 * my_acap.nver);
    std::ifstream in_file(record_file);
    for(int i = 0; i < 9 * my_acap.nver; i++)
    {
        in_file >> feature(i);
    }
    in_file.close();



//    for(int i = 0; i <= N; i++)
//    {
//        std::string output_file = output_folder + std::to_string(i) + ".obj"; // the path to save the deformed mesh
//        double alpha = 0 + double(i) / (N);
//        /// ACAP to vertex
//        my_acap.reconstruction(feature, alpha, output_file);
//    }

//        std::string output_file = output_folder + std::to_string(i) + ".obj"; // the path to save the deformed mesh
//        double alpha = 0 + double(i) / (N);
        /// ACAP to vertex
        my_acap.reconstruction(feature, alpha, output_file);
        remove( record_file.c_str() );

}
