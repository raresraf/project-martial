/*
                       .        ,                      , .
                                                                  .
                                        ,                                 .
                      7                          ,               , . ,
   00100110111111101          ,   11110111011101      . .           , .


                                                          0     1 , . 100 . .


          YES                       .                              NO .
*/

#include <iostream>
#include <string>

using namespace std;
int main() {
  string s;
  cin >> s;
  if (s.find("0000000") != -1 || s.find("1111111") != -1) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
