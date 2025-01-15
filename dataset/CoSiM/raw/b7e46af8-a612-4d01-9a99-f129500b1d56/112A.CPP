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

  string a, b;

  cin >> a >> b;

  for (int i = 0; i < a.length(); i++)

    a[i] = tolower(a[i]);

  for (int i = 0; i < b.length(); i++)

    b[i] = tolower(b[i]);

  if (a == b)

    cout << "0" << endl;

  else if (a < b)

    cout << "-1" << endl;

  else

    cout << "1" << endl;

  return 0;
}