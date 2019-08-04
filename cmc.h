#pragma once


#ifdef _MSC_VER
# pragma warning (disable : 4244)
#endif



#include <stdbool.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <string.h>



typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;

typedef float f32;
typedef double f64;


#ifdef ARYLEN
# undef ARYLEN
#endif
#define ARYLEN(a) (sizeof(a) / sizeof((a)[0]))



#ifdef max
# undef max
#endif
#ifdef min
# undef min
#endif
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))



#define PI 3.14159265358979323846





static const float DEG2RAD = (float)(PI / 180.0);
static const float RAD2DEG = (float)(180.0 / PI);







static bool float_eq(float a, float b)
{
    return fabs(a - b) <= FLT_EPSILON;
}
static bool float_eq_almost(float a, float b)
{
    return fabs(a - b) <= 0.00001;
}



static float fsel(float a, float b, float c)
{
    return a >= 0 ? b : c;
}
static float clamp(float x, float low, float hi)
{
    x = fsel(x - low, x, low);
    return fsel(x - hi, hi, x);
}




static float mix(float a, float b, float w)
{
    return a + (b - a) * w;
}





static float smoothstep(float edge0, float edge1, float x)
{
    // Scale, bias and saturate x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    // Evaluate polynomial
    return x*x*(3 - 2 * x);
}





static float degToRad(float degrees)
{
    float radians = (float)(degrees * DEG2RAD);
    return radians;
}
static float radToDeg(float radians)
{
    float degrees = (float)(radians * RAD2DEG);
    return degrees;
}




static float angleInRad(float x, float y)
{
    return atan2f(y, x);
}





// https://en.wikipedia.org/wiki/Fast_inverse_square_root
static float invSqrt(float x)
{
    union {
        float f;
        uint32_t i;
    } conv;

    float x2;
    const float threehalfs = 1.5F;

    x2 = x * 0.5F;
    conv.f = x;
    conv.i = 0x5F375A86 - (conv.i >> 1);	// what the fuck? 
    conv.f = conv.f * (threehalfs - (x2 * conv.f * conv.f));
    return conv.f;
}














typedef float vec2[2];
typedef float vec3[3];
typedef float vec4[4];
typedef float quat[4];

typedef vec2 mat2[2];
typedef vec3 mat3[3];
typedef vec4 mat4[4];






static vec3 DirectionUnary[3] =
{
    { 1.f, 0.f, 0.f },
    { 0.f, 1.f, 0.f },
    { 0.f, 0.f, 1.f },
};







static void vec2_dup(vec2 r, const vec2 a)
{
    r[0] = a[0];
    r[1] = a[1];
}
static void vec2_neg(vec2 r, const vec2 a)
{
    r[0] = -a[0];
    r[1] = -a[1];
}
static void vec2_add(vec2 r, const vec2 a, const vec2 b)
{
    r[0] = a[0] + b[0];
    r[1] = a[1] + b[1];
}
static void vec2_sub(vec2 r, const vec2 a, const vec2 b)
{
    r[0] = a[0] - b[0];
    r[1] = a[1] - b[1];
}
static void vec2_scale(vec2 r, const vec2 a, float s)
{
    r[0] = a[0] * s;
    r[1] = a[1] * s;
}
static float vec2_dot(const vec2 a, const vec2 b)
{
    return a[0] * b[0] + a[1] * b[1];
}
static void vec2_cross(vec2 r, const vec2 a, const vec2 b)
{
    vec2 t;
    t[0] = a[1] * b[2] - a[2] * b[1];
    t[1] = a[2] * b[0] - a[0] * b[2];
    vec2_dup(r, t);
}
static float vec2_len2(const vec2 a)
{
    return a[0] * a[0] + a[1] * a[1];
}
static float vec2_len(const vec2 a)
{
    return (float)sqrt(vec2_len2(a));
}
static void vec2_norm(vec2 r, const vec2 a)
{
    float s = 1.f / vec2_len(a);
    vec2_scale(r, a, s);
}
static void vec2_interpo(vec2 r, const vec2 a, const vec2 b, float interpol)
{
    vec2 diff;
    vec2_sub(diff, b, a);
    vec2_scale(diff, diff, interpol);
    vec2_add(r, a, diff);
}


















static void vec3_dup(vec3 r, const vec3 a)
{
    r[0] = a[0];
    r[1] = a[1];
    r[2] = a[2];
}
static void vec3_neg(vec3 r, const vec3 a)
{
    r[0] = -a[0];
    r[1] = -a[1];
    r[2] = -a[2];
}
static void vec3_add(vec3 r, const vec3 a, const vec3 b)
{
    r[0] = a[0] + b[0];
    r[1] = a[1] + b[1];
    r[2] = a[2] + b[2];
}
static void vec3_sub(vec3 r, const vec3 a, const vec3 b)
{
    r[0] = a[0] - b[0];
    r[1] = a[1] - b[1];
    r[2] = a[2] - b[2];
}
static void vec3_scale(vec3 r, const vec3 a, float s)
{
    r[0] = a[0] * s;
    r[1] = a[1] * s;
    r[2] = a[2] * s;
}
static float vec3_dot(const vec3 a, const vec3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static void vec3_cross(vec3 r, const vec3 a, const vec3 b)
{
    vec3 t;
    t[0] = a[1] * b[2] - a[2] * b[1];
    t[1] = a[2] * b[0] - a[0] * b[2];
    t[2] = a[0] * b[1] - a[1] * b[0];
    vec3_dup(r, t);
}
static float vec3_len2(const vec3 a)
{
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}
static float vec3_len(const vec3 a)
{
    return (float)sqrt(vec3_len2(a));
}
static void vec3_norm(vec3 r, const vec3 a)
{
	float s = 1.f / vec3_len(a);
	vec3_scale(r, a, s);
}
static void vec3_interpo(vec3 r, const vec3 a, const vec3 b, float interpol)
{
    vec3 diff;
    vec3_sub(diff, b, a);
    vec3_scale(diff, diff, interpol);
    vec3_add(r, a, diff);
}




static bool vec3_eq_almost(const vec3 a, const vec3 b)
{
    for (u32 i = 0; i < 3; ++i)
    {
        if (fabs(a[i] - b[i]) > 0.00001)
        {
            return false;
        }
    }
    return true;
}














// todo test
static void euler_to_dir(vec3 dir, const vec3 euler)
{
    float yaw = euler[0];
    float pitch = euler[1];
    float roll = euler[2];
    dir[0] = -cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll);
    dir[1] = -sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll);
    dir[2] = cos(pitch)*sin(roll);
}

// todo test
static void dir_to_euler(const vec3 dir, vec3 euler)
{
    euler[0] = -atan(dir[1] / dir[2]);
    euler[1] = atan(dir[0] / dir[2]);
    euler[2] = 0;
}
















static void vec4_dup(vec4 r, const vec4 a)
{
    r[0] = a[0];
    r[1] = a[1];
    r[2] = a[2];
    r[3] = a[3];
}
static void vec4_neg(vec4 r, const vec4 a)
{
    r[0] = -a[0];
    r[1] = -a[1];
    r[2] = -a[2];
    r[3] = -a[3];
}
static void vec4_add(vec4 r, const vec4 a, const vec4 b)
{
    r[0] = a[0] + b[0];
    r[1] = a[1] + b[1];
    r[2] = a[2] + b[2];
    r[3] = a[3] + b[3];
}
static void vec4_sub(vec4 r, const vec4 a, const vec4 b)
{
    r[0] = a[0] - b[0];
    r[1] = a[1] - b[1];
    r[2] = a[2] - b[2];
    r[3] = a[3] - b[3];
}
static void vec4_scale(vec4 r, const vec4 a, float s)
{
    r[0] = a[0] * s;
    r[1] = a[1] * s;
    r[2] = a[2] * s;
    r[3] = a[3] * s;
}
static float vec4_dot(const vec4 a, const vec4 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
static void vec4_cross(vec4 r, const vec4 a, const vec4 b)
{
    vec4 t;
    t[0] = a[1] * b[2] - a[2] * b[1];
    t[1] = a[2] * b[0] - a[0] * b[2];
    t[2] = a[0] * b[1] - a[1] * b[0];
    t[3] = 1.f;
    vec4_dup(r, t);
}
static float vec4_len2(const vec4 a)
{
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}
static float vec4_len(const vec4 a)
{
    return (float)sqrt(vec4_len2(a));
}
static void vec4_norm(vec4 r, const vec4 a)
{
    float s = 1.f / vec4_len(a);
    vec4_scale(r, a, s);
}
static void vec4_interpo(vec4 r, const vec4 a, const vec4 b, float interpol)
{
    vec4 diff;
    vec4_sub(diff, b, a);
    vec4_scale(diff, diff, interpol);
    vec4_add(r, a, diff);
}









static void mat2_dup(mat2 r, const mat2 a)
{
    memcpy(r, a, sizeof(mat2));
}
static void mat2_ident(mat2 r)
{
    static const mat2 m =
    {
        1,0,
        0,1,
    };
    mat2_dup(r, m);
}









static void mat3_dup(mat3 r, const mat3 a)
{
    memcpy(r, a, sizeof(mat3));
}
static void mat3_ident(mat3 r)
{
    static const mat3 m =
    {
        1,0,0,
        0,1,0,
        0,0,1,
    };
    mat3_dup(r, m);
}
static void mul_mat3_vec3(vec3 r, const mat3 m, const vec3 v)
{
    vec3 t;
    t[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
    t[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
    t[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
    vec3_dup(r, t);
}
static void mul_vec3_mat3(vec3 r, const vec3 v, const mat3 m)
{
    vec3 t;
    t[0] = v[0] * m[0][0] + v[1] * m[0][1] + v[2] * m[0][2];
    t[1] = v[0] * m[1][0] + v[1] * m[1][1] + v[2] * m[1][2];
    t[2] = v[0] * m[2][0] + v[1] * m[2][1] + v[2] * m[2][2];
    vec3_dup(r, t);
}
static void mat3_mul(mat3 r, const mat3 a, const mat3 b)
{
    mat3 t;
    mul_mat3_vec3(t[0], a, b[0]);
    mul_mat3_vec3(t[1], a, b[1]);
    mul_mat3_vec3(t[2], a, b[2]);
    mat3_dup(r, t);
}



static bool mat3_inverse(mat3 r, const mat3 mat)
{
    mat3 t;

    float det;
    float a = mat[0][0], b = mat[0][1], c = mat[0][2],
          d = mat[1][0], e = mat[1][1], f = mat[1][2],
          g = mat[2][0], h = mat[2][1], i = mat[2][2];

    t[0][0] =   e * i - f * h;
    t[0][1] = -(b * i - h * c);
    t[0][2] =   b * f - e * c;
    t[1][0] = -(d * i - g * f);
    t[1][1] =   a * i - c * g;
    t[1][2] = -(a * f - d * c);
    t[2][0] =   d * h - g * e;
    t[2][1] = -(a * h - g * b);
    t[2][2] =   a * e - b * d;

    det = a * t[0][0] + b * t[1][0] + c * t[2][0];
    if (float_eq(det, 0))
    {
        return false;
    }

    float s = 1.0f / det;
    mat3_dup(r, t);
    r[0][0] *= s; r[0][1] *= s; r[0][2] *= s;
    r[1][0] *= s; r[1][1] *= s; r[1][2] *= s;
    r[2][0] *= s; r[2][1] *= s; r[2][2] *= s;
    return true;
}






// todo test
// https://gamedev.stackexchange.com/questions/22204/from-normal-to-rotation-matrix
static void mat3_from_dir(mat3 r, const vec3 dir)
{
    vec3 left = { 1,0,0 };
    vec3 up = { 0,1,0 };
    // Find a vector in the plane
    vec3 tangent0, tangent1;
    vec3_cross(tangent0, dir, left);
    if (vec3_dot(tangent0, tangent0) < 0.001)
    {
        vec3_cross(tangent0, dir, up);
    }
    vec3_norm(tangent0, tangent0);
    // Find another vector in the plane
    vec3_cross(tangent1, dir, tangent0);
    vec3_norm(tangent1, tangent1);
    // Construct a 3x3 matrix by storing three vectors in the columns of the matrix
    mat3 t =
    {
        { tangent0[0], tangent1[0], dir[0] },
        { tangent0[1], tangent1[1], dir[1] },
        { tangent0[2], tangent1[2], dir[2] },
    };
    mat3_dup(r, t);
}





static void mat3_rotate_axis(mat3 r, const vec3 axis, float angle)
{
    float length2 = vec3_len2(axis);
    if (length2 < FLT_EPSILON)
    {
        mat3_ident(r);
        return;
    }

    vec3 n;
    vec3_scale(n, axis, 1.f / sqrtf(length2));
    float s = sinf(angle);
    float c = cosf(angle);
    float k = 1.f - c;

    float xx = n[0] * n[0] * k + c;
    float yy = n[1] * n[1] * k + c;
    float zz = n[2] * n[2] * k + c;
    float xy = n[0] * n[1] * k;
    float yz = n[1] * n[2] * k;
    float zx = n[2] * n[0] * k;
    float xs = n[0] * s;
    float ys = n[1] * s;
    float zs = n[2] * s;

    r[0][0] = xx;
    r[0][1] = xy + zs;
    r[0][2] = zx - ys;
    r[1][0] = xy - zs;
    r[1][1] = yy;
    r[1][2] = yz + xs;
    r[2][0] = zx + ys;
    r[2][1] = yz - xs;
    r[2][2] = zz;
}

static void mat3_from_euler(mat3 r, const vec3 euler)
{
    mat3 rot[3];
    for (u32 i = 0; i < 3; ++i)
    {
        mat3_rotate_axis(rot[i], DirectionUnary[i], euler[i]);
    }
    mat3 t;
    mat3_mul(t, rot[2], rot[1]);
    mat3_mul(r, t, rot[0]);
}





static void mat3_from_mat4rot(mat3 r, const mat4 m)
{
    vec3_dup(r[0], m[0]);
    vec3_dup(r[1], m[1]);
    vec3_dup(r[2], m[2]);
}

















static void mat4_dup(mat4 r, const mat4 a)
{
    memcpy(r, a, sizeof(mat4));
}
static void mat4_ident(mat4 r)
{
    static const mat4 m =
    {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1,
    };
    mat4_dup(r, m);
}
static void mul_mat4_vec4(vec4 r, const mat4 m, const vec4 v)
{
    vec4 t;
    t[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3];
    t[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3];
    t[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3];
    t[3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3];
    vec4_dup(r, t);
}
static void mul_vec4_mat4(vec4 r, const vec4 v, const mat4 m)
{
    vec4 t;
    t[0] = v[0] * m[0][0] + v[1] * m[0][1] + v[2] * m[0][2] + v[3] * m[0][3];
    t[1] = v[0] * m[1][0] + v[1] * m[1][1] + v[2] * m[1][2] + v[3] * m[1][3];
    t[2] = v[0] * m[2][0] + v[1] * m[2][1] + v[2] * m[2][2] + v[3] * m[2][3];
    t[3] = v[0] * m[3][0] + v[1] * m[3][1] + v[2] * m[3][2] + v[3] * m[3][3];
    vec4_dup(r, t);
}
static void mat4_mul(mat4 r, const mat4 a, const mat4 b)
{
    mat4 t;
    mul_mat4_vec4(t[0], a, b[0]);
    mul_mat4_vec4(t[1], a, b[1]);
    mul_mat4_vec4(t[2], a, b[2]);
    mul_mat4_vec4(t[3], a, b[3]);
    mat4_dup(r, t);
}


static bool mul_mat4_vec3(vec3 r, const mat4 m, const vec3 _v)
{
    vec4 v;
    vec3_dup(v, _v);
    v[3] = 1.0f;
    vec4 t;
    mul_mat4_vec4(t, m, v);
    vec3_scale(r, t, 1.f / t[3]);
    return t[3] > 0;
}
static void mul_mat4_vec3_0(vec3 r, const mat4 m, const vec3 _v)
{
    vec4 v;
    vec3_dup(v, _v);
    v[3] = 0.0f;
    vec4 t;
    mul_mat4_vec4(t, m, v);
    vec3_dup(r, t);
}





static void mat4_scale(mat4 r, const mat4 a, float s)
{
    mat4 t;
    vec4_scale(t[0], a[0], s);
    vec4_scale(t[1], a[1], s);
    vec4_scale(t[2], a[2], s);
    for (u32 i = 0; i < 4; ++i)
    {
        t[3][i] = a[3][i];
    }
    mat4_dup(r, t);
}
static void mat4_scale_aniso(mat4 r, const mat4 a, const vec3 s)
{
    mat4 t;
    vec4_scale(t[0], a[0], s[0]);
    vec4_scale(t[1], a[1], s[1]);
    vec4_scale(t[2], a[2], s[2]);
    for (u32 i = 0; i < 4; ++i)
    {
        t[3][i] = a[3][i];
    }
    mat4_dup(r, t);
}





static void mat4_get_translate(vec3 r, const mat4 a)
{
    r[0] = a[3][0];
    r[1] = a[3][1];
    r[2] = a[3][2];
}
static void mat4_get_scale(vec3 r, const mat4 a)
{
    r[0] = vec4_len(a[0]);
    r[1] = vec4_len(a[1]);
    r[2] = vec4_len(a[2]);
}
static void mat4_get_dir(vec3 r, const mat4 a)
{
    r[0] = a[0][2];
    r[1] = a[1][2];
    r[2] = a[2][2];
}


static void mat4_set_scale(mat4 r, const mat4 a, float s)
{
    vec3 scale0;
    mat4_get_scale(scale0, a);
    vec3 rescale =
    {
        s / scale0[0],
        s / scale0[1],
        s / scale0[2],
    };
    mat4_scale_aniso(r, a, rescale);
}
static void mat4_set_scale_aniso(mat4 r, const mat4 a, const vec3 s)
{
    vec3 scale0;
    mat4_get_scale(scale0, a);
    vec3 rescale =
    {
        s[0] / scale0[0],
        s[1] / scale0[1],
        s[2] / scale0[2],
    };
    mat4_scale_aniso(r, a, rescale);
}




static void mat4_set_translate(mat4 r, const vec3 v)
{
    r[3][0] = v[0];
    r[3][1] = v[1];
    r[3][2] = v[2];
}
static void mat4_from_pos(mat4 r, const vec3 v)
{
    mat4_ident(r);
    mat4_set_translate(r, v);
}
static void mat4_translate(mat4 r, const vec3 v)
{
    r[3][0] += v[0];
    r[3][1] += v[1];
    r[3][2] += v[2];
}
static void mat4_translate_local(mat4 r, const vec3 v)
{
    mat4 t;
    mat4_ident(t);
    mat4_translate(t, v);
    mat4_mul(r, r, t);
}
static void mat4_rotateX(mat4 r, const mat4 a, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4 rot;
    mat4_ident(rot);
    rot[1][1] = c;
    rot[2][2] = c;
    rot[1][2] = s;
    rot[2][1] = -s;
    mat4_mul(r, a, rot);
}
static void mat4_rotateY(mat4 r, const mat4 a, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4 rot;
    mat4_ident(rot);
    rot[0][0] = c;
    rot[2][2] = c;
    rot[0][2] = s;
    rot[2][0] = -s;
    mat4_mul(r, a, rot);
}
static void mat4_rotateZ(mat4 r, const mat4 a, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4 rot;
    mat4_ident(rot);
    rot[0][0] = c;
    rot[1][1] = c;
    rot[0][1] = s;
    rot[1][0] = -s;
    mat4_mul(r, a, rot);
}
static void mat4_rotate_axis(mat4 r, const vec3 axis, float angle)
{
    float length2 = vec3_len2(axis);
    if (length2 < FLT_EPSILON)
    {
        mat4_ident(r);
        return;
    }

    vec3 n;
    vec3_scale(n, axis, 1.f / sqrtf(length2));
    float s = sinf(angle);
    float c = cosf(angle);
    float k = 1.f - c;

    float xx = n[0] * n[0] * k + c;
    float yy = n[1] * n[1] * k + c;
    float zz = n[2] * n[2] * k + c;
    float xy = n[0] * n[1] * k;
    float yz = n[1] * n[2] * k;
    float zx = n[2] * n[0] * k;
    float xs = n[0] * s;
    float ys = n[1] * s;
    float zs = n[2] * s;

    r[0][0] = xx;
    r[0][1] = xy + zs;
    r[0][2] = zx - ys;
    r[0][3] = 0.f;
    r[1][0] = xy - zs;
    r[1][1] = yy;
    r[1][2] = yz + xs;
    r[1][3] = 0.f;
    r[2][0] = zx + ys;
    r[2][1] = yz - xs;
    r[2][2] = zz;
    r[2][3] = 0.f;
    r[3][0] = 0.f;
    r[3][1] = 0.f;
    r[3][2] = 0.f;
    r[3][3] = 1.f;
}






static void mat4_ortho(mat4 r, float left, float right, float bottom, float top, float n, float f)
{
    r[0][0] = 2.0f / (right - left);
    r[0][1] = 0.0f;
    r[0][2] = 0.0f;
    r[0][3] = 0.0f;
    r[1][0] = 0.0f;
    r[1][1] = 2.0f / (top - bottom);
    r[1][2] = 0.0f;
    r[1][3] = 0.0f;
    r[2][0] = 0.0f;
    r[2][1] = 0.0f;
    r[2][2] = -2.0f / (f - n);
    r[2][3] = 0.0f;
    r[3][0] = -(right + left) / (right - left);
    r[3][1] = -(top + bottom) / (top - bottom);
    r[3][2] = -(f + n) / (f - n);
    r[3][3] = 1.0f;
}
static void mat4_frustum(mat4 r, float left, float right, float bottom, float top, float n, float f)
{
    r[0][0] = 2.0f*n / (right - left);
    r[0][1] = 0.0f;
    r[0][2] = 0.0f;
    r[0][3] = 0.0f;
    r[1][0] = 0.0f;
    r[1][1] = 2.0f*n / (top - bottom);
    r[1][2] = 0.0f;
    r[1][3] = 0.0f;
    r[2][0] = (right + left) / (right - left);
    r[2][1] = (top + bottom) / (top - bottom);
    r[2][2] = -(f + n) / (f - n);
    r[2][3] = -1;
    r[3][0] = 0.0f;
    r[3][1] = 0.0f;
    r[3][2] = -2.0f*(f*n) / (f - n);
    r[3][3] = 0.0f;
}




static void mat4_orthonormalize(mat4 r, const mat4 a)
{
    mat4 t;
    vec4_norm(t[0], a[0]);
    vec4_norm(t[1], a[1]);
    vec4_norm(t[2], a[2]);
    mat4_dup(r, t);
}



























static void mat4_viewRH(mat4 r, const vec3 cam, const vec3 xaxis, const vec3 yaxis, const vec3 zaxis)
{
    for (u32 i = 0; i < 3; ++i)
    {
        r[i][0] = xaxis[i];
        r[i][1] = yaxis[i];
        r[i][2] = zaxis[i];
        r[i][3] = 0;
    }
    r[3][0] = -vec3_dot(xaxis, cam);
    r[3][1] = -vec3_dot(yaxis, cam);
    r[3][2] = -vec3_dot(zaxis, cam);
    r[3][3] = 1;
}
static void mat4_lookatRH(mat4 r, const vec3 cam, const vec3 target, const vec3 up)
{
    vec3 xaxis, yaxis, zaxis;

    vec3_sub(zaxis, cam, target);
    vec3_norm(zaxis, zaxis);

    vec3_cross(xaxis, up, zaxis);
    vec3_norm(xaxis, xaxis);

    vec3_cross(yaxis, zaxis, xaxis);

    mat4_viewRH(r, cam, xaxis, yaxis, zaxis);
}
static void mat4_fpsViewRH(mat4 r, const vec3 cam, float pitch, float yaw)
{
    float cosPitch = cos(pitch);
    float sinPitch = sin(pitch);
    float cosYaw = cos(yaw);
    float sinYaw = sin(yaw);

    vec3 xaxis = { cosYaw, 0, -sinYaw };
    vec3 yaxis = { sinYaw * sinPitch, cosPitch, cosYaw * sinPitch };
    vec3 zaxis = { sinYaw * cosPitch, -sinPitch, cosPitch * cosYaw };

    mat4_viewRH(r, cam, xaxis, yaxis, zaxis);
}










static void mat4_perspective(mat4 proj, float fov, float aspect, float znear, float zfar)
{
    float halfW, halfH;
    if (aspect >= 1)
    {
        halfH = tanf(fov * 0.5f) * znear;
        halfW = halfH * aspect;
    }
    else
    {
        halfW = tanf(fov * 0.5f) * znear;
        halfH = halfW / aspect;
    }
    mat4_frustum(proj, -halfW, halfW, -halfH, halfH, znear, zfar);
}
static float viewDistForPrjNormSize(float fov)
{
    return 0.5f / tanf(fov*0.5f);
}






// todo
static void mat4_infinitePerspective(mat4 proj, float fovy, float aspect, float zNear)
{
    float range = (float)tan(fovy / 2.f) * zNear;
    float left = -range * aspect;
    float right = range * aspect;
    float bottom = -range;
    float top = range;

    memset(proj, 0, sizeof(mat4));
    proj[0][0] = (2.f * zNear) / (right - left);
    proj[1][1] = (2.f * zNear) / (top - bottom);
    proj[2][2] = -1.f;
    proj[2][3] = -1.f;
    proj[3][2] = -2.f * zNear;
}







static void zNearFarFromPerspective(const mat4 proj, vec2 z)
{
    z[0] = (2.0f*proj[3][2]) / (2.0f*proj[2][2] - 2.0f);
    z[1] = ((proj[2][2] - 1.0f)*z[0]) / (proj[2][2] + 1.0);
}

static float aspectFromPerspective(const mat4 proj)
{
    float a = proj[1][1] / proj[0][0];
    return a;
}
static float fovFromPerspective(const mat4 proj)
{
    float fov = atan(1.0f / proj[1][1]) * 2.0;
    return fov;
}











static void mat4_transpose(mat4 r, const mat4 a)
{
    mat4 t;
    for (u32 i = 0; i < 4; ++i)
    {
        for (u32 j = 0; j < 4; ++j)
        {
            t[i][j] = a[j][i];
        }
    }
    mat4_dup(r, t);
}

static bool mat4_inverse(mat4 r, const mat4 a)
{
    float s[6];
    float c[6];
    s[0] = a[0][0] * a[1][1] - a[1][0] * a[0][1];
    s[1] = a[0][0] * a[1][2] - a[1][0] * a[0][2];
    s[2] = a[0][0] * a[1][3] - a[1][0] * a[0][3];
    s[3] = a[0][1] * a[1][2] - a[1][1] * a[0][2];
    s[4] = a[0][1] * a[1][3] - a[1][1] * a[0][3];
    s[5] = a[0][2] * a[1][3] - a[1][2] * a[0][3];

    c[0] = a[2][0] * a[3][1] - a[3][0] * a[2][1];
    c[1] = a[2][0] * a[3][2] - a[3][0] * a[2][2];
    c[2] = a[2][0] * a[3][3] - a[3][0] * a[2][3];
    c[3] = a[2][1] * a[3][2] - a[3][1] * a[2][2];
    c[4] = a[2][1] * a[3][3] - a[3][1] * a[2][3];
    c[5] = a[2][2] * a[3][3] - a[3][2] * a[2][3];

    float det = s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0];
    if (float_eq(det, 0))
    {
        return false;
    }
    float idet = 1.0f / det;

    mat4 t;

    t[0][0] = (a[1][1] * c[5] - a[1][2] * c[4] + a[1][3] * c[3]) * idet;
    t[0][1] = (-a[0][1] * c[5] + a[0][2] * c[4] - a[0][3] * c[3]) * idet;
    t[0][2] = (a[3][1] * s[5] - a[3][2] * s[4] + a[3][3] * s[3]) * idet;
    t[0][3] = (-a[2][1] * s[5] + a[2][2] * s[4] - a[2][3] * s[3]) * idet;

    t[1][0] = (-a[1][0] * c[5] + a[1][2] * c[2] - a[1][3] * c[1]) * idet;
    t[1][1] = (a[0][0] * c[5] - a[0][2] * c[2] + a[0][3] * c[1]) * idet;
    t[1][2] = (-a[3][0] * s[5] + a[3][2] * s[2] - a[3][3] * s[1]) * idet;
    t[1][3] = (a[2][0] * s[5] - a[2][2] * s[2] + a[2][3] * s[1]) * idet;

    t[2][0] = (a[1][0] * c[4] - a[1][1] * c[2] + a[1][3] * c[0]) * idet;
    t[2][1] = (-a[0][0] * c[4] + a[0][1] * c[2] - a[0][3] * c[0]) * idet;
    t[2][2] = (a[3][0] * s[4] - a[3][1] * s[2] + a[3][3] * s[0]) * idet;
    t[2][3] = (-a[2][0] * s[4] + a[2][1] * s[2] - a[2][3] * s[0]) * idet;

    t[3][0] = (-a[1][0] * c[3] + a[1][1] * c[1] - a[1][2] * c[0]) * idet;
    t[3][1] = (a[0][0] * c[3] - a[0][1] * c[1] + a[0][2] * c[0]) * idet;
    t[3][2] = (-a[3][0] * s[3] + a[3][1] * s[1] - a[3][2] * s[0]) * idet;
    t[3][3] = (a[2][0] * s[3] - a[2][1] * s[1] + a[2][2] * s[0]) * idet;
    mat4_dup(r, t);
    return true;
}














static void mat4_from_quat(mat4 r, const quat q)
{
    float a = q[3];
    float b = q[0];
    float c = q[1];
    float d = q[2];
    float a2 = a*a;
    float b2 = b*b;
    float c2 = c*c;
    float d2 = d*d;

    r[0][0] = a2 + b2 - c2 - d2;
    r[0][1] = 2.f*(b*c + a*d);
    r[0][2] = 2.f*(b*d - a*c);
    r[0][3] = 0.f;

    r[1][0] = 2 * (b*c - a*d);
    r[1][1] = a2 - b2 + c2 - d2;
    r[1][2] = 2.f*(c*d + a*b);
    r[1][3] = 0.f;

    r[2][0] = 2.f*(b*d + a*c);
    r[2][1] = 2.f*(c*d - a*b);
    r[2][2] = a2 - b2 - c2 + d2;
    r[2][3] = 0.f;

    r[3][0] = r[3][1] = r[3][2] = 0.f;
    r[3][3] = 1.f;
}



static void mat4_from_euler(mat4 r, const vec3 euler)
{
    mat4 rot[3];
    for (u32 i = 0; i < 3; ++i)
    {
        mat4_rotate_axis(rot[i], DirectionUnary[i], euler[i]);
    }
    mat4 t;
    mat4_mul(t, rot[2], rot[1]);
    mat4_mul(r, t, rot[0]);
}
static void mat4_to_euler(const mat4 a, vec3 euler)
{
    mat4 t;
    mat4_orthonormalize(t, a);
    euler[0] = atan2f(t[1][2], t[2][2]);
    euler[1] = atan2f(-t[0][2], sqrtf(t[1][2] * t[1][2] + t[2][2] * t[2][2]));
    euler[2] = atan2f(t[0][1], t[0][0]);
}












static bool mat4_eq_almost(const mat4 a, const mat4 b)
{
    for (u32 i = 0; i < 4; ++i)
    {
        for (u32 j = 0; j < 4; ++j)
        {
            if (fabs(a[i][j] - b[i][j]) > 0.00001)
            {
                return false;
            }
        }
    }
    return true;
}













static void trs_decompose(const mat4 a, vec3 translation, vec3 rotation, vec3 scale)
{
    scale[0] = vec4_len(a[0]);
    scale[1] = vec4_len(a[1]);
    scale[2] = vec4_len(a[2]);

    mat4 t;
    mat4_orthonormalize(t, a);
    rotation[0] = atan2f(t[1][2], t[2][2]);
    rotation[1] = atan2f(-t[0][2], sqrtf(t[1][2] * t[1][2] + t[2][2] * t[2][2]));
    rotation[2] = atan2f(t[0][1], t[0][0]);

    translation[0] = a[3][0];
    translation[1] = a[3][1];
    translation[2] = a[3][2];
}

static void trs_recompose(mat4 r, const vec3 translation, const vec3 rotation, const vec3 scale)
{
    mat4 rot[3];
    for (u32 i = 0; i < 3; ++i)
    {
        mat4_rotate_axis(rot[i], DirectionUnary[i], rotation[i]);
    }
    mat4 t;
    mat4_mul(t, rot[2], rot[1]);
    mat4_mul(t, t, rot[0]);

    float validScale[3];
    for (u32 i = 0; i < 3; ++i)
    {
        if (fabsf(scale[i]) < FLT_EPSILON)
        {
            validScale[i] = 0.001f;
        }
        else
        {
            validScale[i] = scale[i];
        }
    }
    vec4_scale(t[0], t[0], validScale[0]);
    vec4_scale(t[1], t[1], validScale[1]);
    vec4_scale(t[2], t[2], validScale[2]);
    t[3][0] = translation[0];
    t[3][1] = translation[1];
    t[3][2] = translation[2];
    t[3][3] = 1.f;
    mat4_dup(r, t);
}


















static void quat_dup(quat r, const quat a)
{
    vec4_dup(r, a);
}

static void quat_ident(quat r)
{
    quat a = { 0,0,0,1 };
    quat_dup(r, a);
}
static void quat_add(quat r, const quat a, const quat b)
{
    vec4_add(r, a, b);
}
static void quat_sub(quat r, const quat a, const quat b)
{
    vec4_sub(r, a, b);
}
static void quat_mul(quat r, const quat a, const quat b)
{
    quat t;
    t[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
    t[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2];
    t[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0];
    t[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
    quat_dup(r, t);
}
static void quat_from_axis_angle(quat r, const vec3 axis, float angle)
{
    quat t;
    float rad = angle * 0.5f;
    vec3_scale(t, axis, sinf(rad));
    t[3] = cosf(rad);
}
static void quat_from_euler(quat r, const vec3 euler)
{
    double sX = sin(euler[0] * 0.5);
    double cX = cos(euler[0] * 0.5);
    double sY = sin(euler[1] * 0.5);
    double cY = cos(euler[1] * 0.5);
    double sZ = sin(euler[2] * 0.5);
    double cZ = cos(euler[2] * 0.5);
    r[0] = sY * sZ * cX + cY * cZ * sX;
    r[1] = sY * cZ * cX + cY * sZ * sX;
    r[2] = cY * sZ * cX - sY * cZ * sX;
    r[3] = cY * cZ * cX - sY * sZ * sX;
}
static void quat_to_euler(const quat a, vec3 euler)
{
    mat4 t;
    mat4_from_quat(t, a);
    mat4_to_euler(t, euler);
}



static void quat_interpo(quat r, const quat a, const quat b, float factor)
{
    float cosom = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    quat end;
    quat_dup(end, b);
    if (cosom < 0.0f)
    {
        cosom = -cosom;
        end[0] = -end[0];
        end[1] = -end[1];
        end[2] = -end[2];
        end[3] = -end[3];
    }
    float sclp, sclq;
    // 0.0001 : some epsilon
    if ((1.0f - cosom) > 0.0001f)
    {
        float omega, sinom;
        omega = acosf(cosom);
        sinom = sinf(omega);
        sclp = sinf((1.0f - factor) * omega) / sinom;
        sclq = sinf(factor * omega) / sinom;
    }
    else
    {
        sclp = 1.0f - factor;
        sclq = factor;
    }
    r[0] = sclp * a[0] + sclq * end[0];
    r[1] = sclp * a[1] + sclq * end[1];
    r[2] = sclp * a[2] + sclq * end[2];
    r[3] = sclp * a[3] + sclq * end[3];
}




































typedef struct BBox
{
    vec3 min;
    vec3 max;
} BBox;

static BBox bboxEmpty()
{
    BBox a =
    {
        { INFINITY, INFINITY, INFINITY },
        { -INFINITY, -INFINITY, -INFINITY },
    };
    return a;
}
static void bboxExpandPos(BBox* box, const vec3 p)
{
    for (u32 i = 0; i < 3; ++i)
    {
        box->min[i] = min(box->min[i], p[i]);
        box->max[i] = max(box->max[i], p[i]);
    }
}
static void bboxMerge(BBox* box, const BBox* box1)
{
    for (u32 i = 0; i < 3; ++i)
    {
        box->min[i] = min(box->min[i], box1->min[i]);
        box->max[i] = max(box->max[i], box1->max[i]);
    }
}
static void bboxSize(vec3 size, const BBox* box)
{
    vec3_sub(size, box->max, box->min);
}
static void bboxCenter(vec3 center, const BBox* box)
{
    vec3_add(center, box->min, box->max);
    vec3_scale(center, center, 0.5f);
}
static void bboxCorners(vec3 corners[8], const BBox* box)
{
    float l = box->min[0];
    float r = box->max[0];
    float b = box->min[1];
    float t = box->max[1];
    float n = box->min[2];
    float f = box->max[2];
    vec3 _corners[8] =
    {
        { l, b, n },
        { r, b, n },
        { l, t, n },
        { r, t, n },
        { l, b, f },
        { r, b, f },
        { l, t, f },
        { r, t, f },
    };
    memcpy(corners, _corners, sizeof(_corners));
}
static BBox bboxTransform(const BBox* box, const mat4 mat)
{
    BBox box1 = bboxEmpty();
    if ((INFINITY == box->min[0]) ||
        (INFINITY == box->min[1]) ||
        (INFINITY == box->min[2]) ||
        (-INFINITY == box->max[0]) ||
        (-INFINITY == box->max[1]) ||
        (-INFINITY == box->max[2]))
    {
        return box1;
    }
    float l = box->min[0];
    float r = box->max[0];
    float b = box->min[1];
    float t = box->max[1];
    float n = box->min[2];
    float f = box->max[2];
    vec4 corners[8] =
    {
        { l, b, n, 1.0f },
        { r, b, n, 1.0f },
        { l, t, n, 1.0f },
        { r, t, n, 1.0f },
        { l, b, f, 1.0f },
        { r, b, f, 1.0f },
        { l, t, f, 1.0f },
        { r, t, f, 1.0f },
    };
    for (u32 i = 0; i < 8; ++i)
    {
        vec4 c;
        mul_mat4_vec4(c, mat, corners[i]);
        vec3_scale(c, c, 1.f / c[3]);
        bboxExpandPos(&box1, c);
    }
    return box1;
}



static bool aabbIntersect(const BBox* a, const BBox* b)
{
    if ((a->max[0] < b->min[0]) || (a->min[0] > b->max[0])) return false;
    if ((a->max[1] < b->min[1]) || (a->min[1] > b->max[1])) return false;
    if ((a->max[2] < b->min[2]) || (a->min[2] > b->max[2])) return false;
    return true;
}
static bool aabbIntersectPoint(const BBox* a, const vec3 point)
{
    if ((a->max[0] < point[0]) || (a->min[0] > point[0])) return false;
    if ((a->max[1] < point[1]) || (a->min[1] > point[1])) return false;
    if ((a->max[2] < point[2]) || (a->min[2] > point[2])) return false;
    return true;
}











typedef struct Plane
{
    vec3 normal;
    float d;
} Plane;

static Plane planeFrom3Points(const vec3 pt1, const vec3 pt2, const vec3 pt3)
{
    vec3 a, b, normal;
    vec3_sub(a, pt2, pt1);
    vec3_sub(b, pt3, pt1);
    vec3_cross(normal, a, b);
    vec3_norm(normal, normal);
    float d = vec3_dot(normal, pt1);
    Plane plane = { 0 };
    vec3_dup(plane.normal, normal);
    plane.d = d;
    return plane;
}
static void normalizePlane(Plane* plane)
{
    float len = vec3_len(plane->normal);
    vec3_scale(plane->normal, plane->normal, 1.0f / len);
    plane->d /= len;
}
static float signedDistanceToPoint(const Plane* plane, const vec3 pt)
{
    return vec3_dot(plane->normal, pt) + plane->d;
}
static void closestPointOnPlaneToPoint(vec3 cloest, const Plane* plane, const vec3 pt)
{
    vec3 t;
    vec3_scale(t, plane->normal, signedDistanceToPoint(plane, pt));
    vec3_sub(cloest, pt, t);
}

// http://donw.io/post/frustum-point-extraction/
static bool calcPlanesIntersect(vec3 r, const Plane* p0, const Plane* p1, const Plane* p2)
{
    mat3 M;
    M[0][0] = p0->normal[0]; M[1][0] = p0->normal[1]; M[2][0] = p0->normal[2];
    M[0][1] = p1->normal[0]; M[1][1] = p1->normal[1]; M[2][1] = p1->normal[2];
    M[0][2] = p2->normal[0]; M[1][2] = p2->normal[1]; M[2][2] = p2->normal[2];
    // solve the linear system
    // if M is singular the three planes intersect with a line, not a point
    if (!mat3_inverse(M, M))
    {
        return false;
    }
    // transform the distance vector by the inverse to get the intersection point
    r[0] = M[0][0] * -p0->d + M[1][0] * -p1->d + M[2][0] * -p2->d;
    r[1] = M[0][1] * -p0->d + M[1][1] * -p1->d + M[2][1] * -p2->d;
    r[2] = M[0][2] * -p0->d + M[1][2] * -p1->d + M[2][2] * -p2->d;
    return true;
}

















typedef struct Sphere
{
    vec3 center;
    float radius;
} Sphere;

static Sphere sphereNew(const vec3 center, float radius)
{
    Sphere a;
    memcpy(a.center, center, sizeof(vec3));
    a.radius = radius;
    return a;
}


























enum
{
    FrustumPlane_Near,
    FrustumPlane_Far,
    FrustumPlane_Left,
    FrustumPlane_Right,
    FrustumPlane_Top,
    FrustumPlane_Bottom,

    NumFrustumPlanes
};

typedef struct Frustum
{
    Plane planes[NumFrustumPlanes];
} Frustum;


static Frustum frustumNew(const mat4 vp)
{
    Frustum _frus = { 0 };
    Frustum* frus = &_frus;

    // left
    frus->planes[FrustumPlane_Left].normal[0] = vp[0][3] + vp[0][0];
    frus->planes[FrustumPlane_Left].normal[1] = vp[1][3] + vp[1][0];
    frus->planes[FrustumPlane_Left].normal[2] = vp[2][3] + vp[2][0];
    frus->planes[FrustumPlane_Left].d = vp[3][3] + vp[3][0];

    // right
    frus->planes[FrustumPlane_Right].normal[0] = vp[0][3] - vp[0][0];
    frus->planes[FrustumPlane_Right].normal[1] = vp[1][3] - vp[1][0];
    frus->planes[FrustumPlane_Right].normal[2] = vp[2][3] - vp[2][0];
    frus->planes[FrustumPlane_Right].d = vp[3][3] - vp[3][0];

    // bottom
    frus->planes[FrustumPlane_Bottom].normal[0] = vp[0][3] + vp[0][1];
    frus->planes[FrustumPlane_Bottom].normal[1] = vp[1][3] + vp[1][1];
    frus->planes[FrustumPlane_Bottom].normal[2] = vp[2][3] + vp[2][1];
    frus->planes[FrustumPlane_Bottom].d = vp[3][3] + vp[3][1];

    // top
    frus->planes[FrustumPlane_Top].normal[0] = vp[0][3] - vp[0][1];
    frus->planes[FrustumPlane_Top].normal[1] = vp[1][3] - vp[1][1];
    frus->planes[FrustumPlane_Top].normal[2] = vp[2][3] - vp[2][1];
    frus->planes[FrustumPlane_Top].d = vp[3][3] - vp[3][1];

    // near
    frus->planes[FrustumPlane_Near].normal[0] = vp[0][3] + vp[0][2];
    frus->planes[FrustumPlane_Near].normal[1] = vp[1][3] + vp[1][2];
    frus->planes[FrustumPlane_Near].normal[2] = vp[2][3] + vp[2][2];
    frus->planes[FrustumPlane_Near].d = vp[3][3] + vp[3][2];

    // far
    frus->planes[FrustumPlane_Far].normal[0] = vp[0][3] - vp[0][2];
    frus->planes[FrustumPlane_Far].normal[1] = vp[1][3] - vp[1][2];
    frus->planes[FrustumPlane_Far].normal[2] = vp[2][3] - vp[2][2];
    frus->planes[FrustumPlane_Far].d = vp[3][3] - vp[3][2];

    for (u32 i = 0; i < NumFrustumPlanes; ++i)
    {
        normalizePlane(frus->planes + i);
    }
    return *frus;
}





static bool pointInFrustum(const Frustum* frus, const vec3 pt)
{
    for (u32 i = 0; i < NumFrustumPlanes; ++i)
    {
        if (signedDistanceToPoint(frus->planes + i, pt) < 0)
        {
            return false;
        }
    }
    return true;
}
static bool sphereInFrustum(const Frustum* frus, const Sphere* sphere)
{
    for (u32 i = 0; i < NumFrustumPlanes; ++i)
    {
        if (signedDistanceToPoint(frus->planes + i, sphere->center) < -sphere->radius)
        {
            return false;
        }
    }
    return true;
}




static bool aabbInFrustum(const Frustum* frus, const BBox* aabb)
{
    // http://www.txutxi.com/?p=584
    vec3 box[2];
    vec3_dup(box[0], aabb->min);
    vec3_dup(box[1], aabb->max);
    for (u32 i = 0; i < NumFrustumPlanes; ++i)
    {
        const Plane* plane = frus->planes + i;
        int px = (int)(plane->normal[0] > 0.0f);
        int py = (int)(plane->normal[1] > 0.0f);
        int pz = (int)(plane->normal[2] > 0.0f);
        float dp = plane->normal[0] * box[px][0] + plane->normal[1] * box[py][1] + plane->normal[2] * box[pz][2];
        if (dp < -plane->d)
        {
            return false;
        }
    }
    return true;
}







// http://donw.io/post/frustum-point-extraction/
static void calcFrustumPlanesIntersect(vec3 r, const Plane* p0, const Plane* p1, const Plane* p2)
{
    vec3 bxc, cxa, axb;
    vec3_cross(bxc, p1->normal, p2->normal);
    vec3_cross(cxa, p2->normal, p0->normal);
    vec3_cross(axb, p0->normal, p1->normal);
    vec3 t0, t1, t2;
    vec3_scale(t0, bxc, -p0->d);
    vec3_scale(t1, cxa, -p1->d);
    vec3_scale(t2, axb, -p2->d);
    vec3_add(r, t0, t1);
    vec3_add(r, r, t2);
    vec3_scale(r, r, 1 / vec3_dot(p0->normal, bxc));
}

// 8 vertices
static void calcFrustumCorners(vec3* corners, const Frustum* frus)
{
    calcFrustumPlanesIntersect(corners[0], &frus->planes[FrustumPlane_Left], &frus->planes[FrustumPlane_Bottom], &frus->planes[FrustumPlane_Near]);
    calcFrustumPlanesIntersect(corners[1], &frus->planes[FrustumPlane_Right], &frus->planes[FrustumPlane_Bottom], &frus->planes[FrustumPlane_Near]);
    calcFrustumPlanesIntersect(corners[2], &frus->planes[FrustumPlane_Left], &frus->planes[FrustumPlane_Top], &frus->planes[FrustumPlane_Near]);
    calcFrustumPlanesIntersect(corners[3], &frus->planes[FrustumPlane_Right], &frus->planes[FrustumPlane_Top], &frus->planes[FrustumPlane_Near]);
    calcFrustumPlanesIntersect(corners[4], &frus->planes[FrustumPlane_Left], &frus->planes[FrustumPlane_Bottom], &frus->planes[FrustumPlane_Far]);
    calcFrustumPlanesIntersect(corners[5], &frus->planes[FrustumPlane_Right], &frus->planes[FrustumPlane_Bottom], &frus->planes[FrustumPlane_Far]);
    calcFrustumPlanesIntersect(corners[6], &frus->planes[FrustumPlane_Left], &frus->planes[FrustumPlane_Top], &frus->planes[FrustumPlane_Far]);
    calcFrustumPlanesIntersect(corners[7], &frus->planes[FrustumPlane_Right], &frus->planes[FrustumPlane_Top], &frus->planes[FrustumPlane_Far]);
}

// 5 vertices
static void calcFrustumPyramidCorners(vec3* corners, const Frustum* frus)
{
    calcFrustumPlanesIntersect(corners[0], &frus->planes[FrustumPlane_Left], &frus->planes[FrustumPlane_Bottom], &frus->planes[FrustumPlane_Right]);
    calcFrustumPlanesIntersect(corners[1], &frus->planes[FrustumPlane_Left], &frus->planes[FrustumPlane_Bottom], &frus->planes[FrustumPlane_Far]);
    calcFrustumPlanesIntersect(corners[2], &frus->planes[FrustumPlane_Right], &frus->planes[FrustumPlane_Bottom], &frus->planes[FrustumPlane_Far]);
    calcFrustumPlanesIntersect(corners[3], &frus->planes[FrustumPlane_Left], &frus->planes[FrustumPlane_Top], &frus->planes[FrustumPlane_Far]);
    calcFrustumPlanesIntersect(corners[4], &frus->planes[FrustumPlane_Right], &frus->planes[FrustumPlane_Top], &frus->planes[FrustumPlane_Far]);
}



// http://www.yosoygames.com.ar/wp/2016/12/frustum-vs-pyramid-intersection-also-frustum-vs-frustum/
static bool calcFrustumPyramidIntersect(const Frustum* frus, const Frustum* pyramid, const vec3 frusV[8], const vec3 pyramidV[5])
{
    bool intersects = true;
    for (u32 i = 0; i < NumFrustumPlanes; ++i)
    {
        bool isAnyVertexInPositiveSide = false;
        for (int j = 0; j < 5; ++j)
        {
            isAnyVertexInPositiveSide |= signedDistanceToPoint(frus->planes + i, pyramidV[j]) > 0;
        }
        intersects &= isAnyVertexInPositiveSide;
    }
    for (u32 i = 0; i < NumFrustumPlanes; ++i)
    {
        bool isAnyVertexInPositiveSide = false;
        for (int j = 0; j < 8; ++j)
        {
            isAnyVertexInPositiveSide |= signedDistanceToPoint(pyramid->planes + i, frusV[j]) > 0;
        }
        intersects &= isAnyVertexInPositiveSide;
    }
    return intersects;
}



















static void calcTriNormal(vec3 norm, const vec3 p0, const vec3 p1, const vec3 p2)
{
    vec3 sp10, sp20;
    vec3_sub(sp10, p1, p0);
    vec3_sub(sp20, p2, p0);
    vec3_cross(norm, sp10, sp20);
    //float ln = vec3_len(norm);
    //float l10 = vec3_len(sp10);
    //float l20 = vec3_len(sp20);
    vec3_norm(norm, norm);
    //vec3_scale(norm, norm, ln / (l10*l20));
}




static void calcTriTangent(vec3 tangent, vec3 bitangent,
    const vec3 p0, const vec3 p1, const vec3 p2,
    const vec2 uv0, const vec2 uv1, const vec2 uv2)
{
    vec3 edge0, edge1;
    vec3_sub(edge0, p1, p0);
    vec3_sub(edge1, p2, p0);
    vec2 deltaUV0, deltaUV1;
    vec2_sub(deltaUV0, uv1, uv0);
    vec2_sub(deltaUV1, uv2, uv0);
    float r = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1]);

    tangent[0] = (edge0[0] * deltaUV1[1] - edge1[0] * deltaUV0[1]) * r;
    tangent[1] = (edge0[1] * deltaUV1[1] - edge1[1] * deltaUV0[1]) * r;
    tangent[2] = (edge0[2] * deltaUV1[1] - edge1[2] * deltaUV0[1]) * r;
    vec3_norm(tangent, tangent);

    bitangent[0] = (-edge0[0] * deltaUV1[0] + edge1[0] * deltaUV0[0]) * r;
    bitangent[1] = (-edge0[1] * deltaUV1[0] + edge1[1] * deltaUV0[0]) * r;
    bitangent[2] = (-edge0[2] * deltaUV1[0] + edge1[2] * deltaUV0[0]) * r;
    vec3_norm(bitangent, bitangent);
}











// https://github.com/stackgl/ray-aabb-intersection
static bool rayIntersectAABB(const vec3 org, const vec3 dir, const BBox* aabb, float* pDist)
{
    float lo = -INFINITY;
    float hi = +INFINITY;
    for (u32 i = 0; i < 3; ++i)
    {
        if ((-INFINITY == aabb->min[i]) || (INFINITY == aabb->max[i]))
        {
            return false;
        }
        float dimLo = (aabb->min[i] - org[i]) / dir[i];
        float dimHi = (aabb->max[i] - org[i]) / dir[i];

        if (dimLo > dimHi)
        {
            float tmp = dimLo;
            dimLo = dimHi;
            dimHi = tmp;
        }
        if (dimHi < lo || dimLo > hi)
        {
            return false;
        }
        if (dimLo > lo)
        {
            lo = dimLo;
        }
        if (dimHi < hi)
        {
            hi = dimHi;
        }
    }
    if (lo > hi)
    {
        return false;
    }
    else
    {
        *pDist = lo;
    }
    return true;
}




// http://antongerdelan.net/opengl/raycasting.html
// https://github.com/Jam3/camera-picking-ray
// https://www.mkonrad.net/2014/08/07/simple-opengl-object-picking-in-3d.html
static void rayDirFromView(vec3 rayDir, const vec2 screenPos, const mat4 VP)
{
    mat4 invVP;
    mat4_inverse(invVP, VP);
    vec2 xy = { screenPos[0] * 2.f - 1.f, screenPos[1] * 2.f - 1.f };

    vec4 p0 = { xy[0], xy[1], -1, 1 };
    vec4 p1 = { xy[0], xy[1], +1, 1 };
    mul_mat4_vec4(p0, invVP, p0);
    mul_mat4_vec4(p1, invVP, p1);
    vec3_scale(p1, p1, p0[3]);
    vec3_scale(p0, p0, p1[3]);
    vec3_sub(rayDir, p1, p0);
    vec3_norm(rayDir, rayDir);
}






































































































































































