#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int t, num[101];
  while (cin >> t && t != EOF) {
    int sum = 0, total = 0, Ans = 0;
    for (int i = 0; i < t; i++) {
      cin >> num[i];
      sum += num[i];
    }
    sort(num, num + t);
    for (int i = t - 1; i >= 0 && total <= sum / 2; i--) {
      total += num[i];
      Ans++;
    }
    cout << Ans << endl;
  }
  return 0;
}
