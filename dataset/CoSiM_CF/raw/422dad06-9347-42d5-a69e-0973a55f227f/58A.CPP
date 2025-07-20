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
  int a = 0;
  int i = 0;
  while (i < n) {
    if (A[i] == 'h' && a == 0) {
      a = 1;
    } else if (A[i] == 'e' && a == 1) {
      a = 2;
    } else if (A[i] == 'l' && a == 2) {
      a = 3;
    } else if (A[i] == 'l' && a == 3) {
      a = 4;
    } else if (A[i] == 'o' && a == 4) {
      a = 5;
    }
    i++;
  }
  if (a == 5) {
    printf("YES\n");
  } else {
    printf("NO\n");
  }
}

int main() {

  char A[101];
  scanf("%s", A);

  solve(A);

  return 0;
}
