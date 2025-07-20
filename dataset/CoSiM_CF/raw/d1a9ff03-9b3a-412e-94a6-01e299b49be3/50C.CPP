#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;
const int maxm = 1e5 + 10;
struct point {
  long long x, y;
};

int n;
point mat[maxm << 2];
point real[maxm << 2];
int dirx[] = {0, 0, 1, -1};
int diry[] = {1, -1, 0, 0};

long long cross(point a, point b, point c) {
  return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
}

bool cmp(point a, point b) {
  if (a.x != b.x)
    return a.x < b.x;
  return a.y < b.y;
}

bool cmp2(point a, point b) {
  double v = cross(mat[0], a, b);
  if (fabs(v) < 1e-8) {
    return a.x < b.x;
  }
  return v > 0;
}

int cal(point a, point b) { return max(abs(a.x - b.x), abs(a.y - b.y)); }

int main() {
  scanf("%d", &n);
  for (int i = 0, a, b; i < n; i++) {
    scanf("%d%d", &a, &b);
    for (int dir = 0; dir < 4; dir++) {
      mat[i << 2 | dir].x = a + dirx[dir];
      mat[i << 2 | dir].y = b + diry[dir];
    }
  }
  n <<= 2;
  //    for(int i = 0;i<n;i++) {
  //        std::cout<<mat[i].x<<' '<<mat[i].y<<std::endl;
  //    }
  sort(mat, mat + n, cmp);
  sort(mat + 1, mat + n, cmp2);

  int len = 0;
  real[len++] = mat[0];
  real[len++] = mat[1];

  for (int i = 2; i < n; i++) {
    while (len > 2 && cross(real[len - 2], real[len - 1], mat[i]) <= 0) {
      //            std::cout<<"fail:"<<real[len-2].x<<' '<<real[len-2].y<<'
      //            '<<real[len-1].x<<' '<<real[len-1].y<<' '<<mat[i].x<<'
      //            '<<mat[i].y<<std::endl;
      //            std::cout<<"score:"<<cross(real[len-2],real[len-1],mat[i])<<std::endl;
      len--;
    }
    real[len++] = mat[i];
    //        for(int i = 0;i<len;i++) {
    //            std::cout<<real[i].x<<' '<<real[i].y<<std::endl;
    //        }
    //        std::cout<<"----------------------------"<<std::endl;
  }
  real[len] = real[0];
  //    std::cout<<len<<std::endl;

  int ans = 0;
  for (int i = 0; i < len; i++) {
    ans += cal(real[i], real[i + 1]);
  }
  printf("%d\n", ans);
  return 0;
}
