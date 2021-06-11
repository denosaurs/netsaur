[[block]]
struct Matrix {
    size: vec2<f32>;
    numbers: [[stride(4)]] array<f32>;
};

[[group(0), binding(0)]]
var<storage> matrix1: [[access(read)]] Matrix;
[[group(0), binding(1)]]
var<storage> matrix2: [[access(read)]] Matrix;
[[group(0), binding(2)]]
var<storage> resmat: [[access(read_write)]] Matrix;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] gid: vec3<u32>) {
    resmat.size = vec2<f32>(matrix1.size.x, matrix2.size.y);
    var rc: vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));

    var res: f32 = 0.0;
    var i: i32 = 0;
    loop {
        if (i >= i32(matrix1.size.y)) { 
            break;
        }
        var a: i32 = i + rc.x * i32(matrix1.size.y);
        var b: i32 = rc.y + i * i32(matrix2.size.y);
        res = res + f32(matrix1.numbers[a] * matrix2.numbers[b]);        
        i = i + 1;
    }

    var index: i32 = rc.y + rc.x * i32(matrix2.size.y);
    resmat.numbers[index] = res;
}
