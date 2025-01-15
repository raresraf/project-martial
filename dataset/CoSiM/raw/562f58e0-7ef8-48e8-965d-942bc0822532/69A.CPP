#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int num_forces, x = 0, y = 0, z = 0;
  cin >> num_forces;
  while (num_forces--) {
    int fx, fy, fz;
    scanf("%d %d %d", &fx, &fy, &fz);
    x += fx;
    y += fy;
    z += fz;
  }
  if (x == 0 && y == 0 && z == 0)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
