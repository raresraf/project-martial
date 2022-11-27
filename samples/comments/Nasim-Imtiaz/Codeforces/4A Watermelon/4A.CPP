/*
 * 4A Watermelon.cpp
 *
 *  Created on: Apr 29, 2018
 *      Author: NSM
 */

#include <iostream>
using namespace std;
int main() {
  int w;
  cin >> w;
  if (w % 2 == 0 && w != 2)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
