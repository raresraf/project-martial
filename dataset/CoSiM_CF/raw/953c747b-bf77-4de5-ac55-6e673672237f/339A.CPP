#include <iostream>
#include <string.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int l = s.length();
  int i;
  int one = 0, two = 0, three = 0;
  for (i = 0; i < l; i++) {
    //    cout << "main i "<<i<<" ";
    char c = s[i];
    if (c == '1')
      one++;
    else if (c == '2')
      two++;
    else if (c == '3')
      three++;
    // switch(c)
    // {
    //     case '1':
    //     {
    //         cout << i<<" ";
    //     one++;
    //     }
    //     case '2':
    //     {
    //         cout << i<<" ";
    //     two++;
    //     }
    //     case '3':
    //     {
    //         cout <<i<<" ";
    //     three++;
    //     }
    //     default:
    //     continue;
    // }
  }
  for (i = 0; i < one; i++) {
    cout << "1+";
  }
  for (i = 0; i < two; i++)
    cout << "2+";
  for (i = 0; i < three - 1; i++)
    cout << "3+";
  if (three > 0)
    cout << 3;
}
