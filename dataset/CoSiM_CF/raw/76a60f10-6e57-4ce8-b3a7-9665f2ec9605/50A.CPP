#include <iostream>

using std::cin;
using std::cout;

int main(void) {
  int M, N;

  cin >> M >> N;

  // solution 1
  /*if (M > N)
  {
          int extra = (M % 2) ? (N / 2) : 0;
          cout << (M / 2) * N + extra;
  }
  else
  {
          int extra = (N % 2) ? (M / 2) : 0;
          cout << (N / 2) * M + extra;
  }*/

  // solution 2
  cout << M * N / 2;

  return 0;
}