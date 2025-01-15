#include <bits/stdc++.h>
using namespace std;
#define length(a) (int)a.size()
#define all(a) a.begin(), a.end()
#define tolowercase(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define touppercase(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define ctoi(a) (int)a - 48
#define itoc(a) a + '0'
#define MOD 1000000007
#define ll long long
int calc(int a, int b) {
  int sum = 0;
  if (a > b) {
    sum = min(b, a - b);
    a -= 2 * sum;
    b -= sum;
  } else if (a < b) {
    sum = min(a, b - a);
    a -= sum;
    b -= 2 * sum;
  }
  if (a == 0 || b == 0)
    return sum;
  int x = a / 3;
  a -= 3 * x;
  b -= 3 * x;

  sum += 2 * x;
  if ((a != 0 && b != 0) && (a > 1 || b > 1)) {
    sum++;
  }
  return sum;
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  int t;
  cin >> t;
  while (t--) {
    int abc[3];
    cin >> abc[0] >> abc[1] >> abc[2];
    int sum = min(abc[0], min(abc[1], abc[2]));
    abc[0] -= sum;
    abc[1] -= sum;
    abc[2] -= sum;
    if (abc[0] != 0 && abc[1] != 0) {
      sum += calc(abc[0], abc[1]);
    }
    cout << sum << "\n";
  }
}