#include <iostream>
#include <string>
using namespace std;
string ctos(string a) {
  int len, i;
  i = a.length();
  for (len = 0; len < i; len++) {
    if (a[len] <= 'Z' && a[len] >= 'A')
      a[len] = a[len] + 32;
  }
  return a;
}
string delvovel(string a) {
  int len, i, j, k, count;
  string ans;
  len = a.length();
  count = 0;
  for (i = 0; i < len; i++) {
    if (a[i] != 'a' && a[i] != 'e' && a[i] != 'i' && a[i] != 'o' &&
        a[i] != 'u' && a[i] != 'y') {
      ans[count] = a[i];
      count++;
    }
  }
  ans[count] = '\0';
  cout << ans << endl;
  for (i = 0; i < count; i++) {
    cout << '.';
    cout << ans[i];
  }
  //	ans=ans-count;
  cout << endl;
  return ans;
}
/*char * insert(string a)
{
        int i,j,len;
        char *ans;
        len=a.length();
        for(i=0;i<2*len;i++)
        {
                if(i%2==0)
                        ans[i]='.';
                else
                        ans[i]=a[i/2];
        }
        ans[2*len]='\0';
        return ans;
}*/
int main() {
  string test;
  int i, len, j;
  char ans[200];
  cin >> test;
  test = ctos(test);
  test = delvovel(test);
  // cout << test << endl;
  /*	int new_var=0;
          len=test.length();
          for(i=0;i<len;i++)
          {
                  if(test[i]!='\0')
                          new_var++;
          }

          for(i=0;i<2*new_var;i++)
          {
                  j=i/2;
                  if(i%2==0)
                          ans[i]='.';
                  else
                          ans[i]=test[j];
          }
  //	cout << len << endl;
          ans[2*new_var]='\0';
          if(ans[2*new_var-1]=='.')
                  ans[2*new_var-1]='\0';
          cout << ans << endl;
          return 0;*/
}
