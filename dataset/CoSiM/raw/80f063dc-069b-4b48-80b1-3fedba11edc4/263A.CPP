// https://codeforces.com/contest/263/problem/A

/* https://codeforces.com/problemset/problem/263/A */

#include <iostream>

#include <string>

#include <cstdlib>

using namespace std;

int main()

{

  int x = 0, y = 0, input[5][5];

  for (int i = 0; i < 5; i++)

  {

    for (int j = 0; j < 5; j++)

    {

      cin >> input[i][j];

      if (input[i][j] == 1)

      {

        x = i;

        y = j;
      }
    }
  }

  int z = abs(x - 2) + abs(y - 2);

  cout << z;
}