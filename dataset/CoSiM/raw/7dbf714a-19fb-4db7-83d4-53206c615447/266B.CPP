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

  int n, t;

  cin >> n >> t;

  string s;

  cin >> s;

  for (int i = 0; i < t; i++)

  {

    for (int j = 0; j < s.length() - 1; j++)

    {

      if (s[j] == 'B' && s[j + 1] == 'G')

      {

        swap(s[j], s[j + 1]);

        j++;
      }
    }
  }

  cout << s;

  return 0;
}