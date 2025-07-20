#include <iostream>
using namespace std;
int main() {
  int n, k, n1, j = 0;
  int participnt[100];
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> participnt[i];
    if (i + 1 == k)
      n1 = participnt[i];
  }
  for (int i = 0; i < n; i++) {
    if (participnt[i] >= n1 && participnt[i] > 0) {
      j = j + 1;
    }
  }
  cout << j << endl;
  return 0;
}
