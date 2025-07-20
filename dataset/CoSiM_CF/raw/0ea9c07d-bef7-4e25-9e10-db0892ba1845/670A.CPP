/*
 * =====================================================================================
 *
 *       Filename:  670a.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Saturday 09 July 2016 07:04:22  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shounak Dey (), dylandey1996@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  int min, max;
  if ((n + 1) % 7 == 0)
    min = 1 + 2 * (n / 7);
  else
    min = 2 * (n / 7);
  if (n % 7 == 0)
    max = 2 * (n / 7);
  else
    max = 2 + 2 * (n / 7);
  cout << min << " " << max << endl;
  return 0;
}
