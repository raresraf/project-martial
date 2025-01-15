#include <iostream>
using namespace std;

int main() {

  int dn;
  int M, N;
  cin >> M >> N;
  if ((M % 2 == 0) || (N % 2 == 0))
    dn = (M * N) / 2;
  else
    dn = (M - 1) * N / 2 + (N - 1) / 2;
  cout << dn << "\n";
}
