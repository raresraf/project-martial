#include <iostream>

using namespace std;

int main() {

  int M = 0;
  int N = 0;
  int ans = 0;

  cin >> M >> N;

  if (N % 2 == 0) {
    ans = M * (N / 2);
  } else {
    if (N == 1) {
      ans = M / 2;
    } else {
      ans = M * ((N - 1) / 2) + (M / 2);
    }
  }
  cout << endl << ans;

  return 0;
}
