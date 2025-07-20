// https://codeforces.com/contest/69/problem/A

/* https://codeforces.com/problemset/problem/69/A */

#include <iostream>

#include <string>

using namespace std;

int main()

{

  int n;

  cin >> n;

  int x = 0, y = 0, z = 0, x1, y1, z1;

  for (int i = 0; i < n; i++)

  {

    cin >> x1 >> y1 >> z1;

    x = x + x1;

    y = y + y1;

    z = z + z1;
  }

  if ((x == 0) && (y == 0) && (z == 0))

    cout << "YES";

  else

    cout << "NO";
}