/*
                               , . .                                      , , ,
   . n      .                           , . , .


                                                                 n (1   n 1000)
   .         n                                    , 0     1. 1, , . , . .


                                                    , .
*/

#include <iostream>
#include <stdio.h>

using namespace std;

int main() {
  int count, available = 0;
  cin >> count;
  int P, V, T;
  for (int i = 0; i < count; i++) {
    cin >> P >> V >> T;
    if ((P + V + T) >= 2)
      available++;
  }
  cout << available;
  return 0;
}
