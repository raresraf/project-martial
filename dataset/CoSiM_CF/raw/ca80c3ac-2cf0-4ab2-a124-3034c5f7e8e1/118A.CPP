#include <bits/stdc++.h>
typedef long long ll;
typedef double ld;
#define vll vector<ll>
#define vvll vector<vll>
#define vld vector<ld>
#define vvld vector<vld>
#define pll pair<ll, ll>
#define vllp vector<pll>
#define MOD 1000000007
#define endl "\n"
#define MAX 1000000007
#define nl << endl
#define f(i, a, b) for (i = a; i < b; i++)
#define pb push_back
#define mp make_pair
#define F first
#define S second
using namespace std;

int main() {
  string s;
  cin >> s;
  string ans = "";
  for (int i = 0; i < s.length(); ++i) {
    if (s[i] == 'A' || s[i] == 'a' || s[i] == 'E' || s[i] == 'e' ||
        s[i] == 'I' || s[i] == 'i' || s[i] == 'O' || s[i] == 'o' ||
        s[i] == 'U' || s[i] == 'u' || s[i] == 'Y' || s[i] == 'y') {
      continue;
    }
    ans = ans + ".";
    if (s[i] >= 'A' && s[i] <= 'Z') {
      ans = ans + char(s[i] + 32);
    } else
      ans = ans + s[i];
  }
  cout << ans;
}