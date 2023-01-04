#version 330

in vec2 v_uv;
in vec4 v_tint;

out vec3 f_color;

uniform sampler2D albedo;

void main() {
    f_color = texture(albedo, vec2(v_uv.x, -v_uv.y)).rgb * v_tint.rgb;
}