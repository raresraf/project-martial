#include <iostream>

using namespace std;

int main() {
  int M, N;
  cin >> M >> N;

  cout << M * (N / 2) + (N % 2 * M / 2) << endl;
}