#include <bits/stdc++.h>

#define ll long long int

#define mod 1000000007

#define pb push_back

#define mp make_pair

#define ff first

#define ss second

#define all(v) v.begin(), v.end()

using namespace std;

int main()

{

  ios_base::sync_with_stdio(false);

  cin.tie(0);

  string s;

  cin >> s;

  string ans;

  for (int i = 0; i < s.length(); i++)

  {

    s[i] = tolower(s[i]);

    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')

      continue;

    else

    {

      ans.pb('.');

      ans.pb(s[i]);
    }
  }

  cout << ans;

  return 0;
}