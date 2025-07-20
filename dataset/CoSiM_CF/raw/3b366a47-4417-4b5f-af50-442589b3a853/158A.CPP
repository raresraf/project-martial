#include <iostream>
using namespace std;
int main() {
  int n, k, score[100], temp = 0, i, maxi;
  cin >> n >> k;
  for (i = 0; i < n; i++) {
    cin >> score[i];
    if (i + 1 == k) {
      maxi = score[i];
    }
  }
  for (i = 0; i < n; i++) {
    if (score[i] >= maxi && score[i] > 0) {
      temp++;
    }
  }
  cout << temp << endl;
  return 0;
}
