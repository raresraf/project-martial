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

  set<char> d;

  for (auto i : s)

    d.insert(i);

  if (d.size() & 1)

    cout << "IGNORE HIM!";

  else

    cout << "CHAT WITH HER!";

  return 0;
}