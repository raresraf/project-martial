/*
 * =====================================================================================
 *
 *       Filename:  template.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Thursday 21 May 2015 03:51:34  IST
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
  long long int k, n, w;
  cin >> k >> n >> w;
  long long int i, ans = 0;
  ans = (w * (2 * k + (w - 1) * k)) / 2;
  if (ans < n)
    cout << "0" << endl;
  else
    cout << ans - n << endl;
  return 0;
}
