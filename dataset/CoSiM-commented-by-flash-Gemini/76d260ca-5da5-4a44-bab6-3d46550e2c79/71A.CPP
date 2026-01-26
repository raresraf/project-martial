#include <bits/stdc++.h>

#define ll long long int

#define mod 1000000007

using namespace std;

int main()

{

  int n;

  cin >> n;

  while (n--)

  {

    string s;

    cin >> s;

    if (s.length() > 10)

    {

      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;

    }

    else

      cout << s << endl;
  }
}