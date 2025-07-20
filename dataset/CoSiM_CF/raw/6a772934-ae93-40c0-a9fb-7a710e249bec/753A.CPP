#include <bits/stdc++.h>

#define ll long long int

#define mod 1000000007

using namespace std;

int main()

{

  int n, n1;

  cin >> n;

  n1 = n;

  vector<int> a;

  for (int i = 1;; i++)

  {

    if ((n1 - i) >= 0)

    {

      a.push_back(i);

      n1 -= i;

    }

    else

      break;
  }

  if (n1)

  {

    int i = a.size() - 1;

    while (n1)

    {

      a[i] += 1;

      n1--;

      i--;
    }

    cout << a.size() << endl;

    for (int i = 0; i < a.size(); i++)

      cout << a[i] << " ";

    cout << endl;

  }

  else

  {

    cout << a.size() << endl;

    for (int i = 0; i < a.size(); i++)

      cout << a[i] << " ";

    cout << endl;
  }
}