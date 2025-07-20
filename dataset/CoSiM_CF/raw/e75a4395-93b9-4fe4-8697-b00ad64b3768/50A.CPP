#include <bits/stdc++.h>
using namespace std;

int main() {
  int M, N, i, mul;

  cin >> M >> N;

  if (1 <= M && M <= N && N <= 16) {
    mul = (M * N) / 2;
    cout << mul;
  }

  return 0;
}