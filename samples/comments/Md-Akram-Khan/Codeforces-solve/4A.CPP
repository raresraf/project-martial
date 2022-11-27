#include <bits/stdc++.h>
using namespace std;
int main() {
  int t;
  cin >> t;
  while (t--) {
    int n, k;
    cin >> n >> k;
    vector<int> a;
    long long sum = 0;
    if (k <= n) {
      if ((n % 2 == 0 && k % 2 == 0) || (n % 2 == 1 && k % 2 == 1)) {
        for (int i = 1; i < k; i++) {
          sum += 1;
          a.push_back(1);
        }
        a.push_back(n - sum);
        cout << "YES" << endl;
        for (int i = 0; i < a.size(); i++)
          cout << a[i] << " ";
        cout << endl;
      } else if (n % 2 == 0 && k <= n / 2) {
        for (int i = 1; i < k; i++) {
          sum += 2;
          a.push_back(2);
        }
        a.push_back(n - sum);
        cout << "YES" << endl;
        for (int i = 0; i < a.size(); i++)
          cout << a[i] << " ";
        cout << endl;
      } else
        cout << "NO" << endl;
    } else
      cout << "NO" << endl;
  }
  return 0;
}
