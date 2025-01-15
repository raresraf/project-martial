#include <ctype.h>
#include <fstream>
#include <iostream>
using namespace std;

int main() {
  // freopen("input.txt","r",stdin);
  int M, N;
  cin >> M >> N;

  cout << (M * N) / 2;
  return 0;
}