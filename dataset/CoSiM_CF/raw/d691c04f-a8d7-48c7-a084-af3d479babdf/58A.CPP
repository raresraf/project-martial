#include <iostream>
using namespace std;

int flag;
int len;
string s;
void co(int j) {
  for (int i = j; i < len; i++) {
    if (s[i] == 'o') {
      flag = 1;
      //  cout<<"O"<<endl;
      //  cout<<flag<<endl;
      break;
    }
  }
}

void cll(int j) {
  for (int i = j; i < len; i++) {
    if (s[i] == 'l') {
      co(i + 1);

      break;
    }
  }
}

void cl(int j) {
  for (int i = j; i < len; i++) {
    if (s[i] == 'l') {
      cll(i + 1);
      break;
    }
  }
}

void ce(int j) {
  for (int i = j; i < len; i++) {
    if (s[i] == 'e') {
      cl(i + 1);
      break;
    }
  }
}
int main() {
  cin >> s;
  flag = 0;
  len = s.length();
  for (int i = 0; i < len; i++) {
    if (s[i] == 'h') {
      ce(i + 1);
      break;
    }
  }

  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
