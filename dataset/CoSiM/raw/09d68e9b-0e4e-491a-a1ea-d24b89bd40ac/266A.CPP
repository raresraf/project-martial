#include <bits/stdc++.h>
#define endl "\n"
#define lli long long int
#define ulli unsigned long long int

using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  int n;
  cin >> n;

  string s;
  cin >> s;

  int count = 0;
  for (int i = 1; i < s.size() - 1; i++) {
    if (!(s[i] != s[i - 1] && s[i] != s[i + 1])) {
      count++;
      s.erase(s.begin() + i);
      i--;
    }
  }

  if (s[0] == s[1] && s.size() == 2)
    count++;

  cout << count << endl;
}