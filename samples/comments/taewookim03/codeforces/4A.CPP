//
// Created by Taewoo Kim on 2/7/2017.
//

#include <iostream>
using namespace std;

int main() {
  int w;
  cin >> w;
  if (w <= 3)
    cout << "NO";
  else if (w % 2 == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}