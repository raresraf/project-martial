#include <iostream>
using namespace std;
int main() {
  string s, s2;
  cin >> s >> s2;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] >= 65 and s[i] <= 90) {
      s[i] += 32;
    }
    if (s2[i] >= 65 and s2[i] <= 90) {
      s2[i] += 32;
    }
  }
  //	cout<<s<<endl<<s2<<endl;
  int d = 0;
  /*for(int i=0;i<s.length();i++)
  {
          d=s[i]-s2[i];
  }*/
  if (s < s2) {
    d = -1;
  } else if (s > s2) {
    d = 1;
  } else if (s == s2) {
    d = 0;
  }
  cout << d;
}
