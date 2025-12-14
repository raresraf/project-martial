#include <bits/stdc++.h>
using namespace std;
#define rep(i, a, b) for (int i = a; i < b; i++)
typedef long long int ll;
#define pb push_back
#define elif else if
#define mp make_pair
#define F first
#define S second
int main() {
  ll n, k;
  cin >> n >> k;
  string s;
  cin >> s;
  ll len = s.size();
  while (k--) {
    bool okay[n];
    memset(okay, 1, sizeof(okay));
    rep(i, 0, len - 1) {
      if (okay[i] == true) {
        if (s[i] == 'B' && s[i + 1] == 'G') {
          swap(s[i], s[i + 1]);
          //  cout<<"value of i ans i+1 are"<<" "<<i<<" "<<i+1<<endl;
          okay[i + 1] = false;
        }
      }
    }
  }
  cout << s << endl;

  return 0;
}