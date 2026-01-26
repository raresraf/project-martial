#include <iostream>

using namespace std;

int main() {
  int M;
  int N;
  cin >> M;
  cin >> N;
  int area = M * N;
  if (area % 2 != 0) {
    area--;
  }
  cout << area / 2;
  return 0;
}