#include <bits/stdc++.h>

using namespace std;

int strlen(char A[]) {
  int i = 0;
  while (A[i] != '\n' && A[i] != '\0' && A[i] != ' ')
    i++;
  return i;
}

void solve(char A[]) {
  int n = strlen(A);
  char B[1000];
  int c = 0;
  for (int i = 0; i < n; i++) {
    if (A[i] - 'A' >= 0 && A[i] - 'Z' <= 0) {
      A[i] = A[i] - 'A' + 'a';
    }
    if (A[i] != 'a' && A[i] != 'e' && A[i] != 'i' && A[i] != 'o' &&
        A[i] != 'u' && A[i] != 'y') {
      B[c] = '.';
      c++;
      B[c] = A[i];
      c++;
    }
  }
  B[c] = '\0';
  printf("%s\n", B);
}

int main() {

  char A[101];

  scanf("%s", A);

  solve(A);

  return 0;
}
