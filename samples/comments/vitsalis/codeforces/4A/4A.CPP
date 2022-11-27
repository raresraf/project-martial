#include <cstdio>

using namespace std;

int main() {
  int w, num;
  fscanf(stdin, "%d", &w);
  if (w == 2) {
    fprintf(stdout, "NO");
    return 0;
  }
  if (w % 2 == 0) {
    fprintf(stdout, "YES");
  } else {
    fprintf(stdout, "NO");
  }

  return 0;
}
