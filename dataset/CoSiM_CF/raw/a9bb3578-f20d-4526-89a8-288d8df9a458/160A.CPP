#include <bits/stdc++.h>
using namespace std;
#define length(a) (int)a.size()
#define all(a) a.begin(), a.end()
#define tolowercase(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define touppercase(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define lsb(x) (x & (-x))
#define ctoi(a) (int)a - 48
#define itoc(a) a + '0'
#define MOD 1000000007
#define ll long long

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  int n;
  cin >> n;
  int arr[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n);
  int coins = 0, currsum = 0;
  for (int i = n - 1; i >= 0; i--) {
    currsum += arr[i];
    coins++;
    if (currsum * 2 > sum)
      break;
  }
  cout << coins << "\n";
}