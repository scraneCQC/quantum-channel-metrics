
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
u3(4.71238898038469,0,1.57079632679490) q[0];
u3(3.14159265358979,0,3.14159265358979) q[1];
cx q[1],q[0];
u1(1.57079632679490) q[0];
u3(4.71238898038469,4.71238898038469,0) q[2];
cx q[2],q[0];
u1(3.14154389215331) q[0];
u3(4.71238898038469,4.71238898038469,0) q[3];
u3(4.87614364832467e-5,0,4.71238898038469) q[2];
cx q[2],q[0];
u1(1.57079632679490) q[0];
cx q[1],q[0];
u1(1.57079632679490) q[0];
u3(1.57079632679490,4.71238898038469,1.57079632679490) q[2];
cx q[2],q[0];
u1(1.57079632679490) q[0];
u3(1.57079632679490,0,1.57079632679490) q[1];
cx q[0],q[1];
u1(1.57079632679490) q[1];
cx q[3],q[1];
u3(3.14159265358979,0,0.0286920511172770) q[1];
u3(0.0286920511172778,1.57079632679490,1.57079632679490) q[3];
cx q[3],q[1];
u3(4.71238898038469,1.57079632679490,0) q[1];
cx q[0],q[1];
u3(1.57079632679490,0,1.57079632679490) q[0];
cx q[0],q[2];
cx q[2],q[1];
u1(1.57079632679490) q[1];
cx q[3],q[1];
u1(3.11290060247252) q[1];
u3(1.57079632679490,4.71238898038469,1.57079632679490) q[0];
u3(0.0286920511172778,1.57079632679490,1.57079632679490) q[3];
cx q[3],q[1];
u1(1.57079632679490) q[1];
cx q[2],q[1];
cx q[0],q[1];
u1(1.57079632679490) q[1];
cx q[3],q[1];
u1(6.25449325606231) q[1];
u3(1.57079632679490,4.71238898038469,1.57079632679490) q[2];
u3(0.0286920511172778,4.71238898038469,1.57079632679490) q[3];
cx q[3],q[1];
u3(4.71238898038469,1.57079632679490,0) q[1];
cx q[0],q[1];
u3(1.57079632679490,0,1.57079632679490) q[0];
cx q[2],q[0];
u1(1.57079632679490) q[0];
cx q[0],q[1];
u1(1.57079632679490) q[1];
cx q[3],q[1];
u1(3.11290060247252) q[1];
u3(0.0286920511172771,4.71238898038469,4.71238898038469) q[3];
cx q[3],q[1];
u1(1.57079632679490) q[1];
cx q[0],q[1];
cx q[2],q[0];
u3(1.57079632679490,0,3.14159265358979) q[0];
u3(1.57079632679490,0,1.57079632679490) q[2];
cx q[2],q[1];
u1(1.57079632679490) q[1];
cx q[3],q[1];
u3(3.14159265358979,0,3.14164141502628) q[1];
u3(4.87614364832467e-5,4.71238898038469,1.57079632679490) q[3];
cx q[3],q[1];
u3(4.71238898038469,1.57079632679490,0) q[1];
cx q[2],q[1];
u3(1.57079632679490,0,1.57079632679490) q[3];
u3(1.57079632679490,0,3.14159265358979) q[1];
