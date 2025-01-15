#include <cstdio>

using namespace std;

int main() {
  int count = 0;
  int N;
  char s[100][101];

  fscanf(stdin, "%d", &N);

  for (int i = 0; i < N; ++i) {
    fscanf(stdin, "%s", s[i]);
    for (int j = 0; s[i][j] != '\0'; ++j) {
      ++count;
    }
    if (count <= 10) {
      fprintf(stdout, "%s\n", s[i]);
    } else {
      fprintf(stdout, "%c%d%c\n", s[i][0], count - 2, s[i][count - 1]);
    }

    count = 0;
  }
}
