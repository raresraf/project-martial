#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
bool check(unsigned long long int a) {
  int count = 0, temp;
  while (a != 0) {
    temp = a % 10;
    if (temp == 4 || temp == 7)
      count++;
    a /= 10;
  }
  if (count == 4 || count == 7)
    return true;
  return false;
}
int main() {
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  ll n = 1, i, j, t;
  unsigned long long int a;
  cin >> a;
  if (check(a)) {
    cout << "YES";
  } else
    cout << "NO";
  cout << "\n";
  return 0;
}