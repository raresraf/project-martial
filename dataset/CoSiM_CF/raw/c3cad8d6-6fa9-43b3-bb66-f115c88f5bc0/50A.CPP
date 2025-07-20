#include <iostream>
using namespace std;
int main() {
  int M, N;
  cin >> M >> N;
  int size = M * N;
  if (size % 2 == 0)
    cout << size / 2;
  else
    cout << (size - 1) / 2;
  return 0;
}