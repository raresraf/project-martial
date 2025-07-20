#include <iostream>
using namespace std;

main() {
  string str;
  cin >> str;
  int n = str.length();
  int index, flag = 0;
  for (int i = 0; i < n; i++) {
    if (str[i] != '+') {
      char min = str[i];
      // cout<<"min "<<min<<endl;
      for (int j = i + 1; j < n; j++) {
        if (min > str[j] && str[j] != '+') {
          min = str[j];
          index = j;
          flag = 1;
        }
      }
      if (flag == 1) {
        str[index] = str[i];
        str[i] = min;
      }
      // cout<<str<<endl;
    }
    flag = 0;
  }
  cout << str << endl;
}
