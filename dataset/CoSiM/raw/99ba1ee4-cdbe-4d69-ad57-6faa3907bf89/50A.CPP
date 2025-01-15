// thinking:
/**
  if there is one even edge, use 2-edge to fill, it must be filled, so result =
m*n/2 if all edges are odd, think split it a odd*even and a 1*odd, finally a 1*1
left, result = m*n/2
**/
#include <iostream>
using namespace std;
int main() {
  int m, n;
  cin >> m >> n;
  cout << (m * n / 2);
  return 0;
}
