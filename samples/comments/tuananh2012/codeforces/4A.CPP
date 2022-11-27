#include <bits/stdc++.h>
using namespace std;
int t, n, a[1001];
int main() {
  cin >> t;
  while (t--) {
    cin >> n;
    for (int i = 0; i < n; i++) {
      cin >> a[i];
    }
    int suma = a[0], sumb = 0;
    int l = 1, r = n - 1;
    bool turn = 1;
    int eat = a[0];
    int num = 1;
    while (l <= r) {
      num++;
      if (turn) {
        int sum = 0;
        while (sum <= eat && r >= l) {
          sum += a[r];
          r--;
        }
        sumb += sum;
        eat = sum;
      } else {
        int sum = 0;
        while (sum <= eat && l <= r) {
          sum += a[l];
          l++;
        }
        suma += sum;
        eat = sum;
      }
      turn = !turn;
    }
    cout << num << " " << suma << " " << sumb << endl;
  }
}