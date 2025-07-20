#include <algorithm>
#include <bits/stdc++.h>
#include <ctype.h>
using namespace std;
/* Too long
int main() {
        string str,find_me="AEIOUYaeiouy";
        int i;
        cin>> str;
        size_t found=str.find_first_of(find_me);
        while (found != string::npos) {
                str[found]='*';
                found=str.find_first_of(find_me,found+1);
        }
        for (i=0; i<=str.size()-1; ) {
        if (str[i]!='*'){
                i=i+1;
        }
        else {
        str.erase(i,1);
        }
        }

        for (i=0; i<=str.size()-1;) {
                str.insert(i,1,'.');
                i=i+2;
        }
        for (i=0; i<=str.size()-1; i++) {
                if ((str[i]>='A') && (str[i]<='Z')) {
                        str[i]=tolower(str[i]);
                }
        }
        cout << str <<endl;
}
*/
bool vokal(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y');
}
int main() {
  string str;
  int i;
  cin >> str;
  for (i = 0; i <= str.size() - 1; i++) {
    str[i] = tolower(str[i]);
  }
  for (i = 0; i <= str.size() - 1; i++) {
    if (!vokal(str[i])) {
      printf(".%c", str[i]);
    }
  }
  printf("\n");
}
