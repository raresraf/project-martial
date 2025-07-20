#include <iostream>

#include <string.h>

using namespace std;

int main()

{

  int a, n;

  n = 0;

  string c;

  cin >> a;

  for (int i = 0; i < a; i++)

  {

    cin >> c;

    if (c == "++X" || c == "X++")

      n++;

    else

      n--;
  }

  cout << n;
}