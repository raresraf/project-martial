#include <bits/stdc++.h>
using namespace std;
int main() {
  int M, N;
  cin >> M >> N;
  int num = 0;
  int div = M / 2;
  int rem = M % 2;
  num = num + div * N + rem * (N / 2);
  cout << num;
}
