﻿//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 130
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	in vec3 vertexColor;	    // variable input from Attrib Array selected by glBindAttribLocation
	out vec3 color;				// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;



class Triangle {
	unsigned int vao;	// vertex array object id
	//float x1, x2, x3, y1, y2, y3;
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
public:
	Triangle() {
		Animate(0);
	}

	void Create(float x1,float y1, float x2, float y2, float x3, float y3, float xx1, float yy1, float xx2, float yy2, float xx3, float yy3) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { x1, x2, x3, y1, y2, y3 , xx1, xx2, xx3, yy1, yy2, yy3 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),  // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

		// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}
	void Create(float x1, float y1, float x2, float y2, float x3, float y3) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { x1, x2, x3, y1, y2, y3  };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),  // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1; // *sinf(t);
		sy = 1; // *cosf(t);
		wTx = 0; //  4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
	}

	void Draw() {
		mat4 M(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};


class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[5000000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
				
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		if (nVertices >= 500000) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[6 * nVertices] = wVertex.v[0];
		vertexData[6 * nVertices + 1] = wVertex.v[1];
		vertexData[6 * nVertices + 2] = wVertex.v[2];
		vertexData[6 * nVertices + 3] = 1; // red
		vertexData[6* nVertices + 4] = 1; // green
		vertexData[6 * nVertices + 5] = 0; // blue
		nVertices++;
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 6 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}
																														//Ez nem biztos h jól jeleníti meg 
	void AddPoint(float cX, float cY,float cZ) {
		if (nVertices >= 500000) return;

		vec4 wVertex = vec4(cX, cY, cZ, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[6 * nVertices] = wVertex.v[0];
		vertexData[6 * nVertices + 1] = wVertex.v[1];
		vertexData[6 * nVertices + 2] = wVertex.v[2];
		vertexData[6 * nVertices + 3] = 128/256; // red
		vertexData[6* nVertices + 4] =128/256; // green
		vertexData[6 * nVertices + 5] = 128/256; // blue
		nVertices++;
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 6 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}
	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};
Triangle triangle;
LineStrip lineStrip;
LineStrip lineStrip2;
LineStrip lineStrip3;
LineStrip lineStrip4;
LineStrip lineStrip5;
LineStrip lineStrip6;

LineStrip *lines= new LineStrip[12];
class Circle {
	//float center, radius;
public:
	Circle() {};
	Circle(int icenter, int iradius) {
		//center = icenter;
	//	radius = iradius;
	}
	void Create() {

	}
	void Draw(float cx, float cy, float r, int stripes)
	{
		float pi = 3.14;
		float i;
	
		
		
		for (i = 0; i <= stripes; i++) {
			lineStrip.AddPoint(
				cx + (r * cos(i *  2*pi / stripes)),
				cy + (r* sin(i * 2*pi / stripes))
			);
		}
		



	}
	void TriangleFantry() {

		glBegin(GL_TRIANGLE_FAN);
		
		glVertex3f(0.0f, 0.0f, 0.0f); //vertex 1
		glColor3f(1.0f, 0.0f, 1.0f);
		glVertex3f(0.0f, 1.0f, 0.0f); //vertex 2
		glColor3f(1.0f, 0.0f, 1.0f);
		glVertex3f(1.0f, 0.0f, 0.0f); //vertex 3
		glColor3f(1.0f, 0.0f, 1.0f);
		glVertex3f(1.5f, 1.0f, 0.0f); //vertex 4
		glColor3f(1.0f, 0.0f, 1.0f);

		glEnd();

	}
	void DrawFilledCircle(float cx, float cy, float r) // 1-10 iges koordinátákban
	{
		int stripes = 60;
			float pi = 3.14;
			float i;


			glBegin(GL_TRIANGLE_FAN);

			for (i = 0; i <= stripes; i++) {
				glVertex2f(
					cx + (r * cos(i * 2 * pi / stripes)),
					cy + (r* sin(i * 2 * pi / stripes))

				);
				
			}
			glEnd();
	}
	void DrawSphereFromTriangle(float cx=0, float cy=0, float cz=0, float r=0) {

		glBegin(GL_TRIANGLE_FAN);
		glVertex3f(0, 0, 0);
		glVertex3f(5, 3, 2);
		glVertex3f(0, 1, -8);

		glEnd();


	}

};

class SphereUniverse {
	float r;
	/*float theta;
	float fi;*/
	vec4 center; //0,0,0

public:
	 SphereUniverse() {}
	 SphereUniverse(float x, float y, float z,float ir) {
		 center = vec4(x, y, z, 1);
		 r = ir;

	 }
	 void Create() {
		 for (int i = 0; i < 12; i++) {
			  lines[i].Create();                                                                                   //Ezt egy konstruktórban inicializáltam :O
		 }
	 }
	void  Drawline(float thetadeg,float fideg) {
		float theta = 3.14 / 2;  //45 fok
		
		float linesamout = 100;
		float pi = 3.14;
		float fi = 0;
		for (int i = 0; i < linesamout; i++) {
			float theta = (i * 2 * pi / linesamout);                                       //Vízszintesen 
			lineStrip.AddPoint(countx(theta, fi), county(theta, fi), countz(theta, fi));
			
		}
		float theta3 = 23.26*pi/180;
		for (int i = 0; i < linesamout; i++) {
			float fi = (i * 2 * pi / linesamout); 
			//float fi2 = -23.26*pi / 180+(i * 2 * pi / linesamout);					//Vízszintesen 
			lineStrip3.AddPoint(countx(theta3, fi), county(theta3, fi), countz(theta3,fi));

		}
		 theta3 = 40*pi / 180;
		for (int i = 0; i < linesamout; i++) {
			float fi = (i * 2 * pi / linesamout);
			//float fi2 = -23.26*pi / 180+(i * 2 * pi / linesamout);					//Vízszintesen 
			lineStrip4.AddPoint(countx(theta3, fi), county(theta3, fi), countz(theta3, fi));

		}
		float fi3 = 60*pi / 180;
		for (int i = 0; i < linesamout; i++) {
			float theta = (i * 2 * pi / linesamout);                                       //Vízszintesen 
			lineStrip5.AddPoint(countx(theta, fi3), county(theta, fi3), countz(theta, fi3));

		}
		 fi3 = 22*pi / 180;
		for (int i = 0; i < linesamout; i++) {
			float theta = (i * 2 * pi / linesamout);                                       //Vízszintesen 
			lineStrip6.AddPoint(countx(theta, fi3), county(theta, fi3), countz(theta, fi3));

		}
		
		
		
		float theta2 = 3.14 / 2;  
		for (int i = 0; i < linesamout; i++) {
			float fi = (i * 2 * pi / linesamout);                                          //Függőlegesen
			lineStrip2.AddPoint(countx(theta2, fi), county(theta2, fi), countz(theta2, fi));

		}



	 }


	void createGlobe() {
		float linesamout = 100;
		float pi = 3.14;


		float theta = 0;
		for (int i = 0; i < 6; i++) {

			for (int j = 0; j < linesamout; j++) {
				float fi = (j * 2 * pi / linesamout);                                         //Vízszintes
				lines[i].AddPoint(countx(theta, fi), county(theta, fi), countz(theta, fi));
			}
			theta += 60;
		}

		float fi = 0;
		for (int i = 6; i < 12; i++) {

			for (int j = 0; j < linesamout; j++) {
				float theta = (j * 2 * pi / linesamout);                                          //Függőlegesen
				lines[i].AddPoint(countx(theta, fi), county(theta, fi), countz(theta, fi));
			}
			fi += 30;
		}
		


		
	

	}


	void GlobeDisplay() {
		for (int i = 0; i < 12; i++) {
			lines[i].Draw();
		}

	}

	float countx(float theta, float fi) {
		float x = 0;
		x = r*cosf(fi)*sinf(theta);

		return x;
		
	}
	float county(float theta, float fi) {
		
		float z = 0;
		z = r*cos(theta);
		return z;
	}
	float countz(float theta, float fi) {
		float y = 0;
		y = r*sinf(fi)*sinf(theta);
		return y;
	}






};


Circle circ;
SphereUniverse sphere(0,0,0,0.3);
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	//triangle.Create(0,0,3,3,0,5,4,2,3,6,8,7);
	//triangle.Create(0, 0, 3, 3, 0, 5);
	//triangle.Create( 4, 2, 3, 6, 8, 7);
	lineStrip.Create();
	circ.Create();
	lineStrip2.Create();
	lineStrip3.Create();
	lineStrip4.Create();
	lineStrip5.Create();
	lineStrip6.Create();
	sphere.Create();
	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(1, 1, 1, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
//circ.Draw(0.0, 0.0, 1, 60);
//circ.DrawFilledCircle(0.0, 0.0, 10);
//circ.DrawSphereFromTriangle();
	//triangle.Draw();
	sphere.createGlobe();

	lineStrip.Draw();
	lineStrip2.Draw();
	lineStrip3.Draw();
	lineStrip4.Draw();
	lineStrip5.Draw();
	lineStrip6.Draw();
	sphere.GlobeDisplay();
	glutSwapBuffers();									// exchange the two buffers
	
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		lineStrip.AddPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	//triangle.Animate(sec);					// animate the triangle object
	glutPostRedisplay();					// redraw the scene
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

